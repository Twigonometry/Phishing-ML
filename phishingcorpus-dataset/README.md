**IMPORTANT**: this file is not included in the git repository as it is quarantined by windows on download, it's available in my [fork](https://github.com/Twigonometry/MachineLearningPhishing) in case the original goes down. If you do grab the file on a Windows machine, you can add an exemption to Defender. Or you can download the file on an Ubuntu machine

The "phishingcorpus" dataset from Fette et. al's paper 'Learning to Detect Phishing Emails' has been reproduced [here](https://github.com/diegoocampoh/MachineLearningPhishing/blob/master/code/resources/emails-phishing.mbox)

```bash
$ cd phishingcorpus-dataset
$ wget https://github.com/Twigonometry/MachineLearningPhishing/tree/master/code/resources/emails-phishing.mbox
```

It's quite old, so the sophistication of phishing has undoubtedly advanced since then, but it's a start.

If you receive this error when installing spacy packages:

```bash
$ python3 -m spacy download en_core_web_sm
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/mac/.local/lib/python3.8/site-packages/spacy/__main__.py", line 4, in <module>
    setup_cli()
  File "/home/mac/.local/lib/python3.8/site-packages/spacy/cli/_util.py", line 73, in setup_cli
    command = get_command(app)
  File "/home/mac/.local/lib/python3.8/site-packages/typer/main.py", line 350, in get_command
    click_command: click.Command = get_group(typer_instance)
  File "/home/mac/.local/lib/python3.8/site-packages/typer/main.py", line 332, in get_group
    group = get_group_from_info(
  File "/home/mac/.local/lib/python3.8/site-packages/typer/main.py", line 483, in get_group_from_info
    command = get_command_from_info(
  File "/home/mac/.local/lib/python3.8/site-packages/typer/main.py", line 579, in get_command_from_info
    command = cls(
  File "/home/mac/.local/lib/python3.8/site-packages/typer/core.py", line 675, in __init__
    super().__init__(
TypeError: __init__() got an unexpected keyword argument 'no_args_is_help'
```

Run this:

```bash
$ pip3 install click --upgrade
Collecting click
  Downloading click-8.1.3-py3-none-any.whl (96 kB)
     |████████████████████████████████| 96 kB 3.5 MB/s 
Installing collected packages: click
Successfully installed click-8.1.3
mac ~/Documents/PostgradDiss/Phishing-ML/phishingcorpus-dataset git:(main) ✗ python3 -m spacy download en_core_web_sm
Collecting en-core-web-sm==3.5.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl (12.8 MB)
     |████████████████████████████████| 12.8 MB 335 kB/s 

...

Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-3.5.0
✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_sm')
```
