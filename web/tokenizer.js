export class Tokenizer {
    constructor(projectName) {
        //python = '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/
        this.GPT2_SPLIT_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

        //python = '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
        this.GPT4_SPLIT_PATTERN = /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu
        this.active_pattern = null

        this.projectName = projectName;
        this.vocabulary_size = 0;
        this.encoder = {};
        this.decoder = {};
    }

    log(data) {
        console.log(data);
    }

    async load() {
        // load json from a server
        let filename = this.projectName + "/vocabulary.json"
        let response = await fetch(filename);
        let json = await response.json();

        this.vocabulary_size = json.size;
        if (json.pattern_kind == 0) {
            this.active_pattern = this.GPT2_SPLIT_PATTERN;
        } else {
            this.active_pattern = this.GPT4_SPLIT_PATTERN;
        }

        for (const [key, value] of Object.entries(json.vocab)) {
            let k = parseInt(key);
            this.decoder[k] = value;
        }

        for (const c of json.merges) {
            let k = [c.p0, c.p1];
            let v = c.t;
            this.encoder[k] = v;
        }
    }

    findNextEncodingToken(tokens) {
        let value = null;
        let pair = null;

        let iterations = tokens.length - 1;
        if (iterations >= 1) {  
            for (let i = 0; i < iterations; i++) {
                let key = [tokens[i], tokens[i + 1]]  
                if (key in this.encoder) {
                    let token = this.encoder[key];
                    if ((token != null) && ((value == null) || (token < value))) {
                        value = token;
                        pair = key;
                    }
                }
            }
        }

        return [pair, value];
    }

    merge(tokens, pair, value) {
        let processed = [];
        let i = 0;
        while (i < tokens.length) {
            let move_single = true;
            if (i < tokens.length - 1) {
                let current_key = [tokens[i], tokens[i + 1]];
                if (current_key[0] == pair[0] && current_key[1] == pair[1]) {
                    processed.push(value)
                    i += 2
                    move_single = false;
                } 
            }

            if (move_single) {
                processed.push(tokens[i]);
                i += 1;
            }
        }
        return processed;
    }

    encode(text) {
        let tokens = [];
        let utf8encoder = new TextEncoder();
        for (const [current] of text.matchAll(this.active_pattern)) {
            let source_chunk_ids = utf8encoder.encode(current);

            let chunk_ids = []
            for (let v of source_chunk_ids) chunk_ids.push(v);

            while (true) {
                let [pair, value] = this.findNextEncodingToken(chunk_ids);
                if (pair == null)
                    break;

                //this.log(`replacing [${chunk_ids}] ${pair} -> ${value}`);
                chunk_ids = this.merge(chunk_ids, pair, value);
            }

            tokens = tokens.concat(chunk_ids);
        }
        return tokens
    }

    decode(tokens) {
        let utf8 = []

        for (let current of tokens) {
            if (current in this.decoder) {
                utf8 = utf8.concat(this.decoder[current]);
            } else {
                utf8.push(current);
            }
        }

        let utf8decoder = new TextDecoder();
        return utf8decoder.decode(new Uint8Array(utf8));
    }
}
