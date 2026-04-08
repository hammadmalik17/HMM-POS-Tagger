
import pickle
import numpy as np

with open('hmm_model.pkl', 'rb') as f:
    hmm = pickle.load(f)

log_A    = hmm['log_A']
log_B    = hmm['log_B']
log_pi   = hmm['log_pi']
tag2idx  = hmm['tag2idx']
idx2tag  = hmm['idx2tag']
word2idx = hmm['word2idx']
tagset   = hmm['tagset']
T        = len(tagset)

def get_word_idx(word):
    return word2idx.get(word.lower(), word2idx["<UNK>"])

def oov_log_probs(word):
    boost = np.zeros(T)
    w     = word.lower()
    BONUS = 3.0
    def add(tag):
        if tag in tag2idx: boost[tag2idx[tag]] += BONUS
    if w.endswith("ing"):                           add("VBG")
    elif w.endswith("ed"):                          add("VBD"); add("VBN")
    elif w.endswith("ly"):                          add("RB")
    elif w.endswith(("tion","ness","ment","ity")): add("NN")
    elif w.endswith(("er","est")):                  add("JJR"); add("JJS")
    elif word[0].isupper():                         add("NNP")
    elif w.endswith("s"):                           add("NNS"); add("VBZ")
    return boost

def viterbi(sentence):
    N            = len(sentence)
    viterbi_grid = np.full((N, T), -np.inf)
    backpointer  = np.zeros((N, T), dtype=int)
    wi   = get_word_idx(sentence[0])
    emit = log_B[:, wi]
    if wi == word2idx["<UNK>"]: emit = emit + oov_log_probs(sentence[0])
    viterbi_grid[0] = log_pi + emit
    for t in range(1, N):
        wi   = get_word_idx(sentence[t])
        emit = log_B[:, wi]
        if wi == word2idx["<UNK>"]: emit = emit + oov_log_probs(sentence[t])
        for j in range(T):
            scores             = viterbi_grid[t-1] + log_A[:, j]
            bp                 = np.argmax(scores)
            viterbi_grid[t][j] = scores[bp] + emit[j]
            backpointer[t][j]  = bp
    best_last = np.argmax(viterbi_grid[N-1])
    best_path = [best_last]
    for t in range(N-1, 0, -1):
        best_path.append(backpointer[t][best_path[-1]])
    best_path.reverse()
    return [idx2tag[i] for i in best_path], viterbi_grid[N-1][best_last]
