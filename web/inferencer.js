//import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js"
import * as ort  from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/+esm'

//https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/quick-start_onnxruntime-web-script-tag/index_esm.html
//https://github.com/karpathy/llama2.c

export class Inferencer {
    constructor(projectName) {
        this.projectName = projectName;
        this.project = null;
        this.session = null;

    }

    log(data) {
        console.log(data);
    }

    async load() {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        // load json from a server
        let filename = this.projectName + "/project.json"
        let response = await fetch(filename);
        this.project = await response.json();
        //this.log(this.project);

        // load onnx model from a server
        filename = this.projectName + "/ckpt.onnx"

        // specify intra operator threads number to 1 and disable CPU memory arena
        let options = { 
            logSeverityLevel: 2,
            executionProviders: ['wasm', 'cpu']
        };
        this.session = await ort.InferenceSession.create(filename, options);
        const inputNames = this.session.inputNames;
        const outputNames = this.session.outputNames;
        //this.log(`input names: ${inputNames}`);
        //this.log(`output names: ${outputNames}`);
    }

    //
    // forward ids through the model
    //
    async forwardIds(idx) {
        // prepare inputs. a tensor need its corresponding TypedArray as data
        let data = new BigInt64Array(idx.length);
        for (let i = 0; i < idx.length; i++) 
            data[i] = BigInt(idx[i]);

        const input = new ort.Tensor(data, [1, data.length]);

        // prepare feeds. use model input names as keys.
        const feeds = { input: input };

        // feed inputs and run
        const options = { logSeverityLevel: 0 };
        const results = await this.session.run(feeds, options);

        // read from results
        const logits = await results.output.data;
        return Array.from(logits)
    }

    //
    // sampling helpers
    //
    sum(x) {
        let sum = 0.0;
        for (let i = 0; i < x.length; i++) 
            sum += x[i];
        return sum; 
    }

    softmax(x) {
        // find max value (for numerical stability)
        let max = x[0]
        for (let i = 1; i < x.length; i++) 
            if (x[i] > max) 
                max = x[i];
        
        // exp and sum
        let sum = 0.0;
        for (let i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i] - max);
            sum += x[i];
        }
        
        // normalize
        for (let i = 0; i < x.length; i++) 
            x[i] /= sum;
    }

    sampleMult(x, coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        let cdf = 0.0;
        for (let i = 0; i < x.length; i++) {
            cdf += x[i];
            if (coin < cdf) 
                return i;
        }
        return x.length - 1; // in case of rounding errors
    }

    sampleTopp(x, coin, topp) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        let sorted = [];
        const cutoff = (1.0 - topp) / (x.length - 1);
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        for (let i = 0; i < x.length; i++) 
            if (x[i] >= cutoff) 
                sorted.push({index: i, probability: x[i]});
            
        sorted.sort((a, b) => b.probability - a.probability);

        // truncate the list where cumulative probability exceeds topp
        let cumulativeProbability = 0.0;
        let truncated = []
        for (let current of sorted) {
            cumulativeProbability += current.probability;
            truncated.push(current);
            if (cumulativeProbability > topp) 
                break
        }

        // in case of rounding errors, return the full list
        if (truncated.length == 0) 
            truncated = sorted;

        let r = coin * cumulativeProbability;
        // sample from the truncated list
        let cdf = 0.0;
        for (let i = 0; i < truncated.length; i++) {
            cdf += truncated[i].probability;
            if (r < cdf) 
                return truncated[i].index;
        }

        return truncated[truncated.length - 1].index;
    }


    async generate(tokens, maxNewTokens, temperature, topp) {
        let output = [];
        let idx = tokens.slice(0);

        for (let n = 0; n < maxNewTokens; n++) {
            const blockSize = this.project.block_size;
            if (idx.length > blockSize)
                idx = idx.slice(-blockSize);

            let logits = await this.forwardIds(idx); 

            // apply the temperature to the logits
            for (let i = 0; i < logits.length; i++) 
                logits[i] /= temperature;
            
            // apply softmax to the logits to get the probabilities for next token
            this.softmax(logits);

            // flip a (float) coin (this is our source of entropy for sampling)
            const coin = Math.random();

            let next = 0;
            if (topp <= 0.0 || topp >= 1.0) {
                // simply sample from the predicted probability distribution
                next = this.sampleMult(logits, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = this.sampleTopp(logits, coin, topp);
            }

            idx.push(next);
            output.push(next);
        }

        return output;
    }

}
