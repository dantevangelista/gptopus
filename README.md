<p align="center"> 
<img src="https://github.com/dantevangelista/gptopus/blob/main/logo.png">
</p>

GPTopus is a decoder-only transformer that generates new text based on input text.
GPTopus uses `tiktoken`, the OpenAI byte pair encoding (BPE) tokeniser, to encode and decode text.

## 📦 Install
To install required libraries for GPTopus from `pip`:

``` 
$ pip install torch, tiktoken
```

## ⛵ Quickstart
<p align="center"> 
<img src="https://github.com/dantevangelista/gptopus/blob/main/visual/loss.png">
</p>

Figure 1. Training and Validation Loss for 4000 Iterations with `frankestein.txt` Dataset on 15GB TP4 GPU

Using OpenGPT:
* Get input text and place in same folder as `gptopus.py`

eg. `frankenstein.txt`
``` 
Letter 1

_To Mrs. Saville, England._


St. Petersburgh, Dec. 11th, 17—.


You will rejoice to hear that no disaster has accompanied the
commencement of an enterprise which you have regarded with such evil
forebodings. I arrived here yesterday, and my first task is to assure
my dear sister of my welfare and increasing confidence in the success
of my undertaking.

I am already far north of London, and as I walk in the streets of
Petersburgh, I feel a cold northern breeze play upon my cheeks, which
braces my nerves and fills me with delight. Do you understand this
feeling? This breeze, which has travelled from the regions towards
which I am advancing, gives me a foretaste of those icy climes.

```

* Change hyperparameters

eg. 
``` 
max_new_tkns = 1000 # generated text length 
```

* set `filename` as input text

eg.
``` 
filename = 'frankenstein.txt' # set as input text 
```

* Set gpt model

eg.
``` 
enc = tiktoken.encoding_for_model("gpt-4o") # change gpt model 
```

* Generated text file created and placed in same folder as `gptopus.py`,
generated text file will use input filename inserting `_out.txt`

eg. `frankenstein_out.txt`
```
!uing all my family are for your destruction and
str yourself: I felt a strange companions of spirit as far different branches, my gaishown will be the winds, in
a gloom he said, so rapidly. But an
you answer
and I am so sister, Felix describe the lovely ship to this he would that I discovered as your fellow. My kind and warmth of my ryg had
tum, after the fut to others I am not describe; I darted by the
from the advocate of justice of my father whilst he
exception of age of possession to give our comfort on her so and almost trees him, that I distinctly
possible him, I reflect, Victor so dear to humanity of a
soension. It is very pause to fill the scene were extinguish percept me above
in the court days;, but one of the
shel from my father calmed himselfive the fresh weary.
```

## 🪪 License
[MIT](https://github.com/dantevangelista/gptopus/blob/main/LICENSE)