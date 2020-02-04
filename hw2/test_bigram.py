from data import *
import lm


def learn_bigram(data, gamma=0, smooth=1):
    """Learns a bigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    bigram = lm.Bigram(gamma=gamma, smooth=smooth)
    bigram.fit_corpus(data.train)
    print("vocab:", len(bigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", bigram.perplexity(data.train))
    print("dev  :", bigram.perplexity(data.dev))
    print("test :", bigram.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(bigram)
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    return bigram


if __name__ == "__main__":
    dname = "brown"
    # datas = []
    # models = []
    # # Learn the models for each of the domains, and evaluate it
    # for dname in dnames:
    print("-----------------------")
    print(dname)
    data = read_texts("data/corpora.tar.gz", dname)
    # datas.append(data)
    model = learn_bigram(data, gamma=0, smooth=0.001)
    # models.append(model)
    # compute the perplexity of all pairs
    # n = len(dnames)
    # perp_dev = np.zeros((n, n))
    # perp_test = np.zeros((n, n))
    # perp_train = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         perp_dev[i][j] = models[i].perplexity(datas[j].dev)
    #         perp_test[i][j] = models[i].perplexity(datas[j].test)
    #         perp_train[i][j] = models[i].perplexity(datas[j].train)

    # print("-------------------------------")
    # print("x train")
    # print_table(perp_train, dnames, dnames, "table-train.tex")
    # print("-------------------------------")
    # print("x dev")
    # print_table(perp_dev, dnames, dnames, "table-dev.tex")
    # print("-------------------------------")
    # print("x test")
    # print_table(perp_test, dnames, dnames, "table-test.tex")
