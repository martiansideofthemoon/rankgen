import matplotlib.pyplot as plt
import pickle
import numpy as np

setting = ['pg19_gpt2_medium', 'wiki_gpt2_medium', 'pg19_gpt2_xl', 'wiki_gpt2_xl', 'pg19_t5_xxl', 'wiki_t5_xxl', 'pg19_t5_xxl_descartes', 'wiki_t5_xxl_descartes']

for s1 in setting:
    rankers = [
        (f'data_new/greedy/{s1}.tsv.mauve.pkl', 'max_gen_mauve', 'Greedy'),
        (f'data_new/ppl/{s1}.jsonl.mauve.pkl', 'max_gen_mauve', 'PPL-rerank'),
        (f'data_new/t5_xl_inbook_gen_all/{s1}.jsonl.mauve.pkl', 'random_gen_mauve', 'Nucleus'),
        (f'data_new/t5_xl_inbook_gen_all/{s1}.jsonl.mauve.pkl', 'max_gen_mauve', 'RankGen-rerank')
    ]
    hatch_styles = ['x', 'O', 'o', '.']

    all_mauve = []
    plt.rcParams.update({'font.size': 16})
    plt.axis([0.0, 1.0, 0.0, 1.0])

    for i, rr in enumerate(rankers):
        with open(rr[0], 'rb') as f:
            mauve1 = pickle.load(f)[rr[1]]
        all_mauve.append(mauve1)

        plt.plot(mauve1.divergence_curve[:, 0], mauve1.divergence_curve[:, 1])

        if i == 0:
            plt.fill_between(mauve1.divergence_curve[:, 0], mauve1.divergence_curve[:, 1], hatch=hatch_styles[i], label=rr[2], facecolor='white', edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        else:
            prev_mauve = all_mauve[i - 1]
            plt.fill(np.append(prev_mauve.divergence_curve[:, 0], mauve1.divergence_curve[:, 0][::-1]),
                     np.append(prev_mauve.divergence_curve[:, 1], mauve1.divergence_curve[:, 1][::-1]),
                     hatch=hatch_styles[i], label=rr[2], facecolor='white', edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=False, ncol=2)
    plt.title(" ".join(s1.split("_")).replace("descartes", "C4"))
    # plt.legend(loc='upper right')
    plt.xlabel("similarity to Q")
    plt.ylabel("similarity to P")
    plt.savefig(f'{s1}.plot.pdf', bbox_inches="tight")
    plt.clf()
