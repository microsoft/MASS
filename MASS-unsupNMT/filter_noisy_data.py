import re
import argparse

from langdetect import detect
from polyglot.detect import Detector

def get_parser():
    parser = argparse.ArgumentParser(description="Remove noisy data")

    parser.add_argument("--input", type=str,
                        help="The path of input file")
    parser.add_argument("--lang", type=str,
                        help="The language of input file")
    parser.add_argument("--output", type=str, default=None,
                        help="The path of output file")

    return parser

def detect_exist_url(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    url1 = re.findall('http[s]?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls) > 0 or len(url1) > 0

def detect_lang(text, lang):
    try:
        for i, l in enumerate(Detector(text, quiet=True).languages):
            if l.code == lang and i == 0:
                return True
        if detect(text) == lang:
            return True
        return False
    except:
        return False

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    count = 0
    allcount = 0
    f = None
    if args.output is not None:
        f = open(args.output, 'w')
    with open(args.input, encoding='utf-8') as input_file:
        for line in input_file:
            allcount += 1
            line = line.strip()
            if detect_exist_url(line) is False:
                if detect_lang(line, args.lang) is True:
                    count += 1
                    if args.output is not None:
                        f.write(line + '\n')
                #print(line)
            if allcount % 1000000 == 0:
                print("{} sentences processed".format(allcount), count)
    print(count, allcount)

if __name__ == "__main__":
    main()

