import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import pandas as pd

# load model & tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
fname = "Help_emp_w_disability.wav"; duration = 8.35
total_time, segment_length, start_at = librosa.get_duration(path=fname), 20, 0
segment_intervals = range(start_at, int(total_time), segment_length)
num_segments = len(segment_intervals)
full_transcript = ""
for x, start in enumerate(segment_intervals):
    start = x * segment_length + start_at

    # Load segment length (seconds) of the file
    speech, rate = librosa.load(fname, sr=16000, offset=start, duration=segment_length)
    input_values = tokenizer(speech, return_tensors='pt').input_values

    # Get logits from the model
    logits = model(input_values).logits

    # Take argmax value
    pred_ids = torch.argmax(logits, dim=-1)
    
    # Decode into transcript
    transcription = tokenizer.batch_decode(pred_ids)
    full_transcript += " " + transcription[0]
    print(f"Processed segment {x+1}/{num_segments}")

# write transcribed text to a txt file
text_file = open('TiffanyYu_Wav2Vec_Text.txt', 'w')
n = text_file.write(str(full_transcript))
text_file.close()

trans_str = str(full_transcript)

# Function to convert to lowercase, remove symbols & split based on whitespace (' ')
def preprocess(text):
    text = text.lower()
    symbols = '!()-[]{};:"\,<>./?@#$%^&*_~'
    no_symbols = ''
    for x in text:
        if x not in symbols:
            no_symbols += x
    
    text = no_symbols
    text = text.split()
    return text


# Open & read original transcript
fname2 = 'TiffanyYu_Orig.txt'
with open(fname2, encoding = 'utf8') as f:
    original = f.read()

print(original)
original = preprocess(original)
trans_str = preprocess(trans_str)

def print_def(original, trans_str):
    words = pd.DataFrame({'Transcribed Text': trans_str})
    words['Original Text'] = pd.DataFrame({'Original Text': original})
    words['Difference'] = words.apply(lambda x: 'No' if x['Original Text'] == x['Transcribed Text'] else 'Yes', axis = 1)
    words = words[['Original Text', 'Transcribed Text', 'Difference']]

    with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', None,
                        ):
        print(words)

    correctness_df = words.groupby('Difference').count()
    print(correctness_df)
    print('\n')

print_def(original, trans_str)

# Make certain adjustments in Transcribed Text to improve the accuracy
# Track number of corrections
corrections = 0
missing = 0
insertion = 0
false_concat = 0
def add_missing(index, word):
    trans_str.insert(index, word + '(missing)')

def concat(index, word):
    del trans_str[index + 1]
    trans_str[index] = word + '(concat)'

# Add with, so (Missing)
add_missing(18, 'with')
corrections += 1
missing += 1

# Concat percent (per & cent)
concat(original.index('percent'), 'percent')
corrections += 1
false_concat += 1

# Split abillion to a & billion (vs a billion)
index_abillion = trans_str.index('abillion')
trans_str[index_abillion] = 'a (split)'
trans_str.insert(index_abillion + 1, 'billion (split)')
corrections += 1
insertion += 1

# Split untapitential to unta & pitential (vs untapped potential)
index_untapi = trans_str.index("untapitential")
trans_str[index_untapi] = 'unta (split)'
trans_str.insert(index_untapi + 1, 'pitential (split)')
corrections += 1
insertion += 1

# Split armen to arm & en (vs arm and)
index_armen = trans_str.index("armen")
trans_str[index_armen] = 'arm (split)'
trans_str.insert(index_armen + 1, 'en (split)')
corrections += 1
insertion += 1

# Split intect to in tect (vs and tech)
index_untapi = trans_str.index("intect")
trans_str[index_untapi] = 'in (split)'
trans_str.insert(index_untapi + 1, 'tect (split)')
corrections += 1
insertion += 1

# Concat over & all to overall 
concat(original.index('overall'), 'overall')
corrections += 1
false_concat += 1

# Add with (missing)
add_missing(158, 'with')
corrections += 1
missing += 1

# Concat trac & gety to tracgety (vs tragedy)
concat(original.index('tragedy'), 'tragedy')
corrections += 1
false_concat += 1

# Concat us & awle to usawle (vs disabled)
concat(304, 'usawle')
corrections += 1
false_concat += 1

# Concat peo & ple to people
concat(305, 'people')
corrections += 1
false_concat += 1

# concat any & one
concat(322, 'anyone')
corrections += 1
false_concat += 1

# concat a & suming
concat(344, 'asuming')
corrections += 1
false_concat += 1

# add to (missing)
add_missing(372, 'to')
corrections += 1
missing += 1

# concat all & so
concat(435, 'allso')
corrections += 1
false_concat += 1

# add we'll (missing)
add_missing(446, "we'll")
corrections += 1
missing += 1

# Concat some & one to someone (vs someone's)
concat(500, 'someone')
corrections += 1
false_concat += 1

# Concat eighty & ight to eightyight (vs adhd)
concat(508, 'eightyight')
corrections += 1
false_concat += 1

# Concat sixty & two to sixtytwo
concat(514, 'sixtytwo')
corrections += 1
false_concat += 1

# Concat per & cent
concat(515, 'percent')
corrections += 1
false_concat += 1

# Concat work & places
concat(550, 'workplaces')
corrections += 1
false_concat += 1

# Concat cli & ns
concat(568, 'clins')
corrections += 1
false_concat += 1

# Split newhires to new & hires
index_newhires = 601
trans_str[index_newhires] = 'new (split)'
trans_str.insert(index_newhires + 1, 'hires (split)')
corrections += 1
insertion += 1

# Concat work & stations
concat(609, 'workstations')
corrections += 1
false_concat += 1

# Concat colleague & 's
concat(636, "colleague's")
corrections += 1
false_concat += 1

# Concat pro & actively
concat(680, 'proactively')
corrections += 1
false_concat += 1

# Split thereare to there & are
index_there = 714
trans_str[index_there] = 'there (split)'
trans_str.insert(index_there + 1, 'are (split)')
corrections += 1
insertion += 1

# Concat in & formation
concat(730, 'information')
corrections += 1
false_concat += 1

# Add with (missing)
add_missing(840, 'with')
corrections += 1
missing += 1

# concat audio & books
concat(850, 'audiobooks')
corrections += 1
false_concat += 1

# concat tooth & brushes
concat(857, 'toothbrushes')
corrections += 1
false_concat += 1

# Add with (missing)
add_missing(886, 'with')
corrections += 1
missing += 1

# Concat m & ost to most
concat(903, 'most')
corrections += 1
false_concat += 1

# Split unto to un & to
index_unto = 924
trans_str[index_unto] = 'un (split)'
trans_str.insert(index_unto + 1, 'to (split)')
corrections += 1
insertion += 1

# Add need (missing)
add_missing(952, 'need')
corrections += 1
missing += 1

print('After Corrections:')
print_def(original, trans_str)
print('\nTotal number of corrections made: ' + str(corrections))
print('\nTotal number of deletions : ' + str(missing))
print('\nTotal number of insertions : ' + str(insertion))
print('\nTotal number of substitutions : ' + str(false_concat + (109 - missing - insertion)))
print('\nWord Error Rate (WER): ' + str((missing + insertion + false_concat + (109 - missing - insertion)) / 1000))


'''Text Summarization'''


from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def LexRank(text):
    # Set up the parser(s) and tokenizer
    parser = PlaintextParser.from_file(text, Tokenizer("english"))
    # Set up the LexRank summarizer
    summarizer = LexRankSummarizer()
    # Set the number of sentences to include in the summary & include stopwords
    summary_length = 3

    # Generate the summaries
    summary = summarizer(parser.document, summary_length)

    # Print the Original summary
    print('\nLex Rank Summary \n')
    for sentence in summary:
        print(str(sentence) + '\n')

LexRank('TiffanyYu_Orig.txt')

def LSA(text):
    # Set up the parser(s) and tokenizer
    parser = PlaintextParser.from_file(text, Tokenizer("english"))
     # Set up the LexRank summarizer
    summarizer = LsaSummarizer()
    # Set the number of sentences to include in the summary & include stopwords
    summary_length = 3
    # Generate the summaries
    summary = summarizer(parser.document, summary_length)

    # Print the Original summary
    print('\nLSA Summary \n')
    for sentence in summary:
        print(str(sentence) + '\n')

LSA('TiffanyYu_Orig.txt')

def TextRank(text):
    # Set up the parser(s) and tokenizer
    parser = PlaintextParser.from_file(text, Tokenizer("english"))
     # Set up the LexRank summarizer
    summarizer = TextRankSummarizer()
    # Set the number of sentences to include in the summary & include stopwords
    summary_length = 3
    # Generate the summaries
    summary = summarizer(parser.document, summary_length)

    # Print the Original summary
    print('\nTextRank Summary \n')
    for sentence in summary:
        print(str(sentence) + '\n')

TextRank('TiffanyYu_Orig.txt')
