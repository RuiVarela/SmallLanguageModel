# Small Language Model
A minimal gpt training environment

As a research demo I've built [Eça](https://eca.demanda.pt/) persona, a simple gpt model trained from scratch on Eça de Queirós books,   
Eça was a well known Portuguese writer, so this persona rambles only in Português from Portugal.   
The model is tiny compared with LLM's so it gets wierd very quickly.   


https://github.com/RuiVarela/SmallLanguageModel/assets/11543973/0707cd0e-760d-4dc3-b8e4-0588dc8b6873


# Features
- basic tokenizer trainer
- a simple gpt model based on Andrej Karpathy lectures
- model trainer
- python inferencer 
- python sampler
- onnx model export
- Web demo located on `web`
- javascript compatible tokenizer
- javascript model inferencer using onnx
- javascript model sampler

# Development
```bash
python3 -m venv .venv
source .venv/bin/activate

pip3 install regex torch matplotlib 

# export needs
pip3 install onnx onnxruntime
```

# Train a new model
```
python main.py --train_tokenizer
python main.py --tokenize_data
python main.py --train
python main.py --complete "hello world"
```

use the parameter `--project` to specify a working project, by default it will use `tinyshakespeare`
```
python main.py --project "eca_queiroz" --train_tokenizer
python main.py --project "eca_queiroz" --tokenize_message "os marinheiros encontraram o tesouro."
python main.py --project "eca_queiroz" --tokenize_data
python main.py --project "eca_queiroz" --train
python main.py --project "eca_queiroz" --complete "o mundo estava escuro" --temperature 0.3
python main.py --project "eca_queiroz" --export
```

# Web runner
```
# create a new folder under web with `ckpt.onnx` `project.json` and `vocabulary.json`
# update `app.js` to use your newly created folder

# run a local webserver i.e:
cd web
npx light-server -s . -p 8080
```

# Credits
- God [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Texto integral das Obras de Eça de Queirós](http://figaro.fis.uc.pt/queiros/lista_obras.html)
