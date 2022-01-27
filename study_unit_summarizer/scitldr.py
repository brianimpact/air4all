import os
import time
import warnings
import argparse
from tqdm import tqdm
import sys
import torch
from fairseq.models.bart import BARTModel

from utils.transcript_cleansing import fix_misrecognition, inverse_abbreviate


def tldr_summarization(batch_size, data_path, out_path, checkpoint_dir, checkpoint_name,
                       beam, lenpen, max_len_b, min_len, no_repeat_ngram_size):
    # DEFINE MODEL (SCITLDR)
    model = BARTModel.from_pretrained(
        model_name_or_path=checkpoint_dir,
        checkpoint_file=checkpoint_name,
        task='translation'
    )
    if torch.cuda.is_available():
        model.cuda()
        model.half() # TODO: REMOVE AFTER TESTING
    model.eval()

    # CHECK FOR FILES THAT ARE ALREADY PROCESSED
    source_names = [x.replace('.txt', '') for x in os.listdir(data_path)]
    processed_fnames = [x.replace('.out', '') for x in os.listdir(out_path)]
    fnames = []
    for f in source_names:
        if f not in processed_fnames:
            fnames.append(f + '.txt')
    
    # LOAD DATA AND START SUMMARIZING
    for source_file in tqdm(sorted(fnames, key=lambda x: int(x.split(')')[0]))):
        source_fname = os.path.join(data_path, source_file)
        pred_fname = os.path.join(out_path, source_file.replace('.txt', '.out'))
        with open(source_fname, encoding="utf-8") as source, open(pred_fname, 'w', encoding="utf-8") as fout:
            source_text = source.readlines()
            for source_line in source_text:
                # DATA FORMAT: INDEX AND TEXT SEPARATED BY TAB IN EVERY LINE
                source_line = source_line.split('\t')[1]
                # TRANSCRIPT ERROR FIX
                # CHECK FOR WORDS WITH SAME METAPHONE AS THE WORDS IN LABEL (SU) NAME
                source_line = fix_misrecognition(source_file, source_line)
                # CONVERT ABBREVIATED WORDS INTO THEIR FULL PHRASE
                source_line = inverse_abbreviate(source_file, source_line)
                source_segments = []
                # SLIDING WINDOW TO RETAIN INFORMATION ON DOCUMENTS WITH LONGER THAN 1,024 (MAX TOKENS FOR SCITLDR) TOKENS
                for sliding_window in [1024, 512, 256]:
                    text_line = source_line
                    while True:
                        segment = ' '.join(text_line.split()[:sliding_window])
                        # ADDING SPECIAL TOKENS OF SCITLDR (<|TLDR|> .) AT THE END OF EACH SUMMARIZED SEQUENCE (WINDOW)
                        if sliding_window == 1024:
                            try:
                                source_segments.append(model.decode(model.encode(segment)[:-9]) + ' <|TLDR|> .')
                            except IndexError:
                                break
                        else:
                            source_segments.append(segment + ' <|TLDR|> .')
                        text_line = ' '.join(text_line.split()[sliding_window // 2:])
                        if model.encode(text_line).size()[0] < sliding_window or len(source_segments) > 256:
                            break
                # SUMMARIZATION
                summarized = []
                minibatches = [source_segments[i * batch_size: (i + 1) * batch_size] for i in range(len(source_segments // batch_size))]
                if len(source_segments) % batch_size != 0:
                    minibatches.append(source_segments[source_segments // batch_size * batch_size:])
                for minibatch in minibatches:
                    with torch.no_grad():
                        out = model.sample(minibatch, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=no_repeat_ngram_size)
                        summarized.extend(out)
                # WRITE RESULTS
                fout.write('\n'.join([summary.replace('.', '').strip() + '.' for summary in summarized]) + '\n\n')
                fout.flush()


if __name__=='__main__':
    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data_path', default='/data5/air4all/contents', help='Path to data')
    parser.add_argument('--out_path', default='/data5/assets/jinhyun95/air4all/single_doc_summ/v220126', help='Path to output')
    parser.add_argument('--checkpoint_dir', default='/home/jinhyun95/AIRoadmap4All/models/single_doc/checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_name', default='scitldr_catts.tldr-ao.pt', help='Name of the checkpoint')
    parser.add_argument('--beam', default=2, type=int)
    parser.add_argument('--lenpen', default=0.2, type=float)
    parser.add_argument('--max_len_b', default=50, type=int)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    args = parser.parse_args()

    start = time.time()

    # check for paths
    if not os.path.exists(args.data_path):
        print('DATA PATH %s DOES NOT EXIST' % args.data_path)
        exit()
    if args.data_path.endswith('/'):
        args.data_path = args.data_path[:-1]
    if not os.path.exists(args.out_path):
        print('OUTPUT PATH %s DOES NOT EXIST / CREATING OUTPUT PATH' % args.out_path)
        os.makedirs(args.out_path)
    if not os.path.exists(args.checkpoint_dir):
        print('CKPT PATH %s DOES NOT EXIST' % args.checkpoint_dir)
        exit()

    tldr_summarization(**vars(args))

    print('COMPLETE: TOOK %d SECONDS TO RUN THE CODE' % int(time.time() - start))
