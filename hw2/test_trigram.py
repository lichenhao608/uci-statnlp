from data import *
import lm


def learn_trigram(data, print_info=False, sample_sentence=False,
                  gamma=0, smooth=1, lambda1=0.65, lambda2=0.25):
    """Learns a trigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    trigram = lm.Trigram(gamma=gamma, smooth=smooth,
                         lambda1=lambda1, lambda2=lambda2)
    trigram.fit_corpus(data.train)
    if print_info:
        print("vocab:", len(trigram.vocab()))
        # evaluate on train, test, and dev
        print("train:", trigram.perplexity(data.train))
        print("dev  :", trigram.perplexity(data.dev))
        print("test :", trigram.perplexity(data.test))

    if sample_sentence:
        from generator import Sampler
        sampler = Sampler(trigram)
        print("sample: ", " ".join(str(x)
                                   for x in sampler.sample_sentence([])))
        print("sample: ", " ".join(str(x)
                                   for x in sampler.sample_sentence([])))
        print("sample: ", " ".join(str(x)
                                   for x in sampler.sample_sentence([])))

    return trigram


if __name__ == "__main__":
    dnames = ["brown", "reuters", "gutenberg"]
    datas = []
    models = []
    # Learn the models for each of the domains, and evaluate it
    for dname in dnames:
        print("-----------------------")
        print(dname)
        data = read_texts("data/corpora.tar.gz", dname)
        datas.append(data)
        model = learn_trigram(data, gamma=0, smooth=0.0001,
                              print_info=True)
        models.append(model)
    # compute the perplexity of all pairs
    n = len(dnames)
    perp_dev = np.zeros((n, n))
    perp_test = np.zeros((n, n))
    perp_train = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            perp_dev[i][j] = models[i].perplexity(datas[j].dev)
            perp_test[i][j] = models[i].perplexity(datas[j].test)
            perp_train[i][j] = models[i].perplexity(datas[j].train)

    print("-------------------------------")
    print("x train")
    print_table(perp_train, dnames, dnames, "table-train.tex")
    print("-------------------------------")
    print("x dev")
    print_table(perp_dev, dnames, dnames, "table-dev.tex")
    print("-------------------------------")
    print("x test")
    print_table(perp_test, dnames, dnames, "table-test.tex")
