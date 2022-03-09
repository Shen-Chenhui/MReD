'''append paragraph of reviews into the positions of paragraphs of longest review'''
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import time

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def reorder_by_cos(corpus):
    # Arguments:
    # corpus: List[List[paragraphs]]

    # Find longest review
    rev_lens = []
    for rev in corpus:
        rev_len = 0
        for para in rev:
            rev_len += len(para.split(" "))
        rev_lens.append(rev_len)
    longest_rev_id = np.argmax(np.array(rev_lens))
    longest_rev = corpus[longest_rev_id]
    longest_rev_emb = embedder.encode(longest_rev, convert_to_tensor=True)

    rev_match = []
    for rev_id, rev in enumerate(corpus):
        if rev_id == longest_rev_id:
            rev_match.append([])
            continue
        para_match = []
        for para in rev:
            para_emb = embedder.encode(para, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(para_emb, longest_rev_emb)[0]
            cos_scores = cos_scores.cpu()
            # print("cos_scores:")
            # print(cos_scores.size())
            # print(cos_scores)
            top_result = torch.topk(cos_scores, k=1)[1][0]
            para_match.append(top_result)
        rev_match.append(para_match)

    backbone = [[para] for para in longest_rev]

    for rev_id, rev in enumerate(corpus):
        if rev_id == longest_rev_id:
            continue
        for para_id, match_id in enumerate(rev_match[rev_id]):
            backbone[match_id].append(rev[para_id])

    reordered = [para for sec in backbone for para in sec]

    return reordered

# corpus = [['A gorilla is playing drums.',
#           'A lion is running behind its prey.',
#           'A man is eating banana.',
#           'A man is riding a black donkey on an enclosed ground.',
#           ],
#           ['A man is eating food.',
#           'A man is eating a piece of bread.',
#           'The girl is carrying a baby.',
#           'A man is riding a horse.',
#           'A woman is playing violin.',
#           'Two men pushed carts through the woods.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'A cheetah is running behind its prey.'
#           ],
#           ['A wonman is eating food.',
#            'A girl is eating a piece of bread.',
#            'The woman is carrying a baby.',
#            'A man is riding a donkey.'
#            ]
#           ]
#
# reorder_by_cos(corpus)

file_id_list = ["val","test","train"]
for fid in file_id_list:
    sourcef = open(fid + "_concat.source", 'r')
    reorderf = open(fid + "_merge.source", 'w')
    print("processing:", fid)

    start_time = time.time()
    for i, line in enumerate(sourcef):
        if i % 50 == 0:
            current_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
            print(f"Processing line {i}, time {current_time}")
        list_of_reviews = line.split(" <REVBREAK> ")
        num_reviews = len(list_of_reviews)
        list_of_paragraphs = []
        for review in list_of_reviews:
            item = review.rstrip('\n').split(" <sep> ") # get paragraph list for each review
            item = [x for x in item if x.strip(" ")!=""] # prevent tokenizer error
            list_of_paragraphs.append(item)

        reordered = reorder_by_cos(list_of_paragraphs)
        reorderf.write(' '.join(reordered) + "\n")
    sourcef.close()
    reorderf.close()
