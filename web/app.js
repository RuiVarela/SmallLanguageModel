import { Inferencer } from "./inferencer.js"
import { Tokenizer } from "./tokenizer.js"

//
// configuration
//
const project = "eca_queiroz"
const prompt = "A casa estava pintada de um lindo verde"

const newTokensPerRound = 2;
const temperature = 0.45;
const topp = 0.9;

const rounds = -1;
const maxEncodedSize = 1024 * 2;

const sleepTime = 250;


//
// helper methods
//
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function setText(text) {
    document.getElementById("output").innerHTML = text;   
}

function appendText(text) {
    document.getElementById("output").innerHTML += text;   


    const delta = window.scrollY + window.innerHeight - document.body.offsetHeight;
    const threshold = 0;
    if (delta > threshold)
        window.scrollTo(0, document.body.scrollHeight);

    console.log(`delta ${delta} `);
}

//
// load tokenizer and inferencer
//

let tokenizer = new Tokenizer(project)
let inferencer = new Inferencer(project)


setText("A acordar a persona ...")

await Promise.all([
    tokenizer.load(), 
    inferencer.load()
])

//
// test tokenizer
//
if (false) {
    let text = "os marinheiros encontraram o tesouro."
    text = "o sapo não lava o pé porque não quer."
    let encoded = tokenizer.encode(text)
    let decoded = tokenizer.decode(encoded)

    console.log(`text:${text} encoded: [${encoded}]`)
    console.log(`text:${text == decoded} decoded: [${decoded}]`)
}

//console.log("prompt: ", prompt);
setText(prompt)

let currentRound = 0;
var encoded = tokenizer.encode(prompt);

while (true) {
    let generated = await inferencer.generate(encoded, newTokensPerRound, temperature, topp);

    let decoded = tokenizer.decode(generated);
    encoded = encoded.concat(generated);
    if (encoded.length > maxEncodedSize)
        encoded = encoded.slice(-maxEncodedSize);

    //console.log(decoded);
    appendText(decoded);
    await sleep(sleepTime);

    if (rounds > 0 && currentRound++ >= rounds)
        break;

    currentRound++;
}