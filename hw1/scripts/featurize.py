'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meters_feature(text, freq):
    return float(freq['meters'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_call_feature(text, freq):
    return int(freq['call'])
def freq_free_feature(text, freq):
    return int(freq['free'])
def freq_text_feature(text, freq):
    return int(freq['text'])
def freq_offer_feature(text, freq):
    return int(freq['offer'])
def freq_win_feature(text, freq):
    return int(freq['win'])


# def ratio_spam_words(text, freq):
#     spam_words = ['free', 'call', 'win', 'cash', 'offer', '!', 'text']
#     count = sum(freq[word] for word in spam_words if word in freq)
#     return int(count / len(text.split()) if len(text.split()) > 0 else 0)

# def ratio_special_characters(text, freq):
#     special_chars = sum(text.count(c) for c in "!$%&")
#     return int(special_chars / len(text) if len(text) > 0 else 0)

def email_length_feature(text, freq):
    return len(text)

# def capitalized_words_count_feature(text, freq):
#     words = text.split()
#     return sum(1 for word in words if word.isupper())
def freq_http_feature(text, freq):
    return int(freq['http'])
def freq_www_feature(text, freq):
    return int(freq['www'])
def freq_com_feature(text, freq):
    return int(freq['com'])
def freq_href_feature(text, freq):
    return int(freq['href'])
def freq_src_feature(text, freq):
    return int(freq['scr'])
def freq_img_feature(text, freq):
    return int(freq['img'])
def freq_font_feature(text, freq):
    return int(freq['font'])
def freq_div_feature(text, freq):
    return int(freq['div'])
def freq_viagra_feature(text, freq):
    return int(freq['viagra'])
def freq_cash_feature(text, freq):
    return int(freq['cash'])
def freq_urgent_feature(text, freq):
    return int(freq['urgent'])
def freq_deal_feature(text, freq):
    return int(freq['deal'])
def freq_click_feature(text, freq):
    return int(freq['click'])
def freq_pills_feature(text, freq):
    return int(freq['pills'])
def freq_00_feature(text, freq):
    return int(freq['00'])
def freq_td_feature(text, freq):
    return int(freq['td'])
def freq_nbsp_feature(text, freq):
    return int(freq['nbsp'])
def freq_company_feature(text, freq):
    return int(freq['company'])
def freq_statements_feature(text, freq):
    return int(freq['statements'])
def freq_subject_feature(text, freq):
    return int(freq['subject'])
def freq_enron_feature(text, freq):
    return int(freq['enron'])
def freq_hou_feature(text, freq):
    return int(freq['hou'])
def freq_ect_feature(text, freq):
    return int(freq['ect'])
def freq_please_feature(text, freq):
    return int(freq['please'])
def freq_cc_feature(text, freq):
    return int(freq['cc'])
def freq_am_feature(text, freq):
    return int(freq['am'])
def freq_pm_feature(text, freq):
    return int(freq['pm'])
def freq_thanks_feature(text, freq):
    return int(freq['thanks'])
# Count occurrences of suspicious SPAM keywords


# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meters_feature(text, freq))
    # feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    # feature.append(freq_call_feature(text, freq))
    # feature.append(freq_free_feature(text, freq))
    # feature.append(freq_text_feature(text, freq))
    # feature.append(freq_offer_feature(text, freq))
    # feature.append(freq_win_feature(text, freq))
    feature.append(freq_http_feature(text, freq))
    feature.append(freq_www_feature(text, freq))
    feature.append(freq_com_feature(text, freq))
    feature.append(freq_href_feature(text, freq))
    feature.append(freq_src_feature(text, freq))
    feature.append(freq_img_feature(text, freq))
    feature.append(freq_font_feature(text, freq))
    feature.append(freq_div_feature(text, freq))   
    # feature.append(freq_http_feature(text, freq))
    # feature.append(ratio_spam_words(text, freq))
    # feature.append(ratio_special_characters(text, freq))

    # feature.append(email_length_feature(text, freq))
    # feature.append(capitalized_words_count_feature(text, freq))
    # feature.append(count_links_feature(text, freq))  # Counts "http", "www", "com"
    # feature.append(count_html_tags_feature(text, freq))  # Counts HTML tags like "href", "src", "img"
    # feature.append(count_spam_keywords_feature(text, freq))  # Counts suspicious SPAM-specific keywords
    
    feature.append(freq_viagra_feature(text, freq))

    feature.append(freq_cash_feature(text, freq))
    # feature.append(freq_urgent_feature(text, freq))
    # feature.append(freq_deal_feature(text, freq))
    feature.append(freq_click_feature(text, freq))
    feature.append(freq_pills_feature(text, freq))
    feature.append(freq_td_feature(text, freq))
    feature.append(freq_nbsp_feature(text, freq))
    feature.append(freq_company_feature(text, freq))
    feature.append(freq_statements_feature(text, freq))
    feature.append(freq_subject_feature(text, freq))
    feature.append(freq_enron_feature(text, freq))
    feature.append(freq_hou_feature(text, freq))
    feature.append(freq_ect_feature(text, freq))
    feature.append(freq_please_feature(text, freq))
    feature.append(freq_cc_feature(text, freq))      
    feature.append(freq_am_feature(text, freq)) 
    feature.append(freq_pm_feature(text, freq)) 
    feature.append(freq_thanks_feature(text, freq))
    # feature.append(freq_00_feature(text, freq))
    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'test-spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
