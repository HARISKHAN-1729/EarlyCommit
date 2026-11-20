# checkpoints/

Put the two fine-tuned checkpoint zips here:

    checkpoints/laya-earlycommit.zip        Laya fine-tuned without augmentation
    checkpoints/laya-earlycommit-aug.zip    Laya fine-tuned with train-pool augmentation

Then unpack them (the notebooks do this for you):

    python -m earlycommit.checkpoints --zip checkpoints/laya-earlycommit.zip     --name laya-earlycommit
    python -m earlycommit.checkpoints --zip checkpoints/laya-earlycommit-aug.zip --name laya-earlycommit-aug

Result:

    checkpoints/laya-earlycommit/       rl_agent_config.json, model.safetensors, encoder/, tokenizer/
    checkpoints/laya-earlycommit-aug/   same files

`training.ipynb` writes both folders directly if you train the models yourself.
