"""
Trains or scores an NER model.

Will attempt to guess the appropriate word vector file if none is
specified, and will use the charlms specified in the resources
for a given dataset or language if possible.

Example command line:
  python3 -m stanza.utils.training.run_ner.py hu_combined

This script expects the prepared data to be in
  data/ner/{lang}_{dataset}.train.json, {lang}_{dataset}.dev.json, {lang}_{dataset}.test.json

If those files don't exist, it will make an attempt to rebuild them
using the prepare_ner_dataset script.  However, this will fail if the
data is not already downloaded.  More information on where to find
most of the datasets online is in that script.  Some of the datasets
have licenses which must be agreed to, so no attempt is made to
automatically download the data.
"""

import glob
import logging
import os

from stanza.models import ner_tagger
from stanza.utils.datasets.ner import prepare_ner_dataset
from stanza.utils.training import common
from stanza.utils.training.common import Mode

from stanza.resources.prepare_resources import default_charlms, ner_charlms
from stanza.resources.common import DEFAULT_MODEL_DIR

# extra arguments specific to a particular dataset -- ignore Vietnamese NER specific
DATASET_EXTRA_ARGS = {
    "vi_vlsp": [ "--dropout", "0.6",
                 "--word_dropout", "0.1",
                 "--locked_dropout", "0.1",
                 "--char_dropout", "0.1" ],
}

logger = logging.getLogger('stanza') #set up the logger to log train results

def add_ner_args(parser): ##add character language model if there is one, defaults to none
    #add data saying charlm is none by default to pre-existing list of arguments
    parser.add_argument('--charlm', default=None, type=str, help='Which charlm to run on.  Will use the default charlm for this language/model if not set.  Set to None to turn off charlm for languages with a default charlm')

def find_charlm(direction, language, charlm): ##the script attempts to search for a charlm
    #look in saved models to see if the language from the corpus has a charlm
    saved_path = 'saved_models/charlm/{}_{}_{}_charlm.pt'.format(language, charlm, direction)
    if os.path.exists(saved_path):
        #if it does, you should use this model and return the path to this model
        logger.info(f'Using model {saved_path} for {direction} charlm')
        return saved_path

    #look and see if the default model directory contains a charlm for this language, then return this path
    resource_path = '{}/{}/{}_charlm/{}.pt'.format(DEFAULT_MODEL_DIR, language, direction, charlm)
    if os.path.exists(resource_path):
        logger.info(f'Using model {resource_path} for {direction} charlm')
        return resource_path

    #otherwise, there isn't a charlm for this model
    raise FileNotFoundError(f"Cannot find {direction} charlm in either {saved_path} or {resource_path}")

def find_wordvec_pretrain(language):
    ######################################pre-existing note below######################################################
    # TODO: try to extract/remember the specific pretrain for the given model
    # That would be a good way to archive which pretrains are used for which NER models, anyway
    ######################################pre-existing note above######################################################
    pretrain_path = '{}/{}/pretrain/*.pt'.format(DEFAULT_MODEL_DIR, language)
    pretrains = glob.glob(pretrain_path)
    if len(pretrains) == 0:
        raise FileNotFoundError(f"Cannot find any pretrains in {pretrain_path}  Try 'stanza.download(\"{language}\")' to get a default pretrain or use --wordvec_pretrain_path to specify a .pt file to use")
    if len(pretrains) > 1:
        raise FileNotFoundError(f"Too many pretrains to choose from in {pretrain_path}  Must specify an exact path to a --wordvec_pretrain_file")
    pretrain = pretrains[0]
    logger.info(f"Using pretrain found in {pretrain}  To use a different pretrain, specify --wordvec_pretrain_file")
    return pretrain

# Technically NER datasets are not necessarily treebanks
# (usually not, in fact)
# However, to keep the naming consistent, we leave the
# method which does the training as run_treebank
# TODO: rename treebank -> dataset everywhere
def run_treebank(mode, paths, treebank, short_name,
                 temp_output_file, command_args, extra_args):
    ner_dir = paths["NER_DATA_DIR"] ##directory for the NER datafiles is here
    language, dataset = short_name.split("_") ##language and dataset from corpus --> hi_fire2013 = hi, fire2013

    #train, dev, and test files located inside fo NER_DATA_DIR with shortname (hi_fire2013) .{train/dev/test}.json
    train_file = os.path.join(ner_dir, f"{short_name}.train.json")
    dev_file   = os.path.join(ner_dir, f"{short_name}.dev.json")
    test_file  = os.path.join(ner_dir, f"{short_name}.test.json")

    #raise error if they can't find files using shortname -- this is invalid corpus arg has been passed
    if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
        logger.warning(f"The data for {short_name} is missing or incomplete.  Attempting to rebuild...")
        try:
            #try preparing the data if it wasn't found
            prepare_ner_dataset.main(short_name)
        except:
            #otherwise raise error
            logger.error(f"Unable to build the data.  Please correctly build the files in {train_file}, {dev_file}, {test_file} and then try again.")
            raise

    #charlm is either the charlm found in dictionary if exists, or none
    default_charlm = default_charlms.get(language, None)
    #if there is a specific charlm in nercharlms, find that --> all in prepare_resources.py file
    specific_charlm = ner_charlms.get(language, {}).get(dataset, None)
    #if the charlm arg exists, then follow control path
    if command_args.charlm:
        charlm = command_args.charlm
        #if explicitly written to none, it's like the default
        if charlm == 'None':
            charlm = None
    #if there is a specific charlm, set to this
    elif specific_charlm:
        charlm = specific_charlm
    elif default_charlm:
        charlm = default_charlm
    #no charlm arg means default to none
    else:
        charlm = None

    if charlm:
        #if there is a charlm, set up the necessary arguments around it
        #storing the path to the character language models
        forward = find_charlm('forward', language, charlm)
        backward = find_charlm('backward', language, charlm)
        #set up the relevant charlm args if charlm is not None
        charlm_args = ['--charlm',
                       '--charlm_shorthand', f'{language}_{charlm}',
                       '--charlm_forward_file', forward,
                       '--charlm_backward_file', backward]
    else:
        charlm_args = []

    if mode == Mode.TRAIN:
        # VI example arguments:
        #   --wordvec_pretrain_file ~/stanza_resources/vi/pretrain/vtb.pt
        #   --train_file data/ner/vi_vlsp.train.json
        #   --eval_file data/ner/vi_vlsp.dev.json
        #   --lang vi
        #   --shorthand vi_vlsp
        #   --mode train
        #   --charlm --charlm_shorthand vi_conll17
        #   --dropout 0.6 --word_dropout 0.1 --locked_dropout 0.1 --char_dropout 0.1

        #find the default arguments based on the language + dataset for us using the shortname
        #lang is hi for hi_fire2013 ... etc.
        #vietnamese example
        dataset_args = DATASET_EXTRA_ARGS.get(short_name, [])

        #set up the training arguments for the tagger
        #take the train/dev file locations found
        train_args = ['--train_file', train_file,
                      '--eval_file', dev_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'train']
        train_args = train_args + charlm_args + dataset_args + extra_args
        if '--wordvec_pretrain_file' not in train_args:
            # will throw an error if the pretrain can't be found
            wordvec_pretrain = find_wordvec_pretrain(language)
            train_args = train_args + ['--wordvec_pretrain_file', wordvec_pretrain]
        logger.info("Running train step with args: {}".format(train_args))
        ner_tagger.main(train_args)

    if mode == Mode.SCORE_DEV or mode == Mode.TRAIN:
        dev_args = ['--eval_file', dev_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'predict']
        dev_args = dev_args + charlm_args + extra_args
        logger.info("Running dev step with args: {}".format(dev_args))
        ner_tagger.main(dev_args)

    if mode == Mode.SCORE_TEST or mode == Mode.TRAIN:
        test_args = ['--eval_file', test_file,
                      '--lang', language,
                      '--shorthand', short_name,
                      '--mode', 'predict']
        test_args = test_args + charlm_args + extra_args
        logger.info("Running test step with args: {}".format(test_args))
        ner_tagger.main(test_args)


def main():
    common.main(run_treebank, "ner", "nertagger", add_ner_args)

if __name__ == "__main__":
    main()

