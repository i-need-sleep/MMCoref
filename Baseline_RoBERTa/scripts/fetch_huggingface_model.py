import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', type=str)
model_checkpoint = parser.parse_args().src
temp = model_checkpoint.split('/')
if len(temp) == 1:
    save_path = temp[0] + '_'
else:
    save_path = '-'.join(model_checkpoint.split('/'))
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
AutoTokenizer.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
import os
os.rename(save_path, save_path[:-1])