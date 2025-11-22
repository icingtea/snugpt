import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('universal_tagset')


def _to_wordnet_pos(tag: str):
    if tag == "ADJ":
        return wordnet.ADJ
    if tag == "VERB":
        return wordnet.VERB
    if tag == "NOUN":
        return wordnet.NOUN
    if tag == "ADV":
        return wordnet.ADV
    return wordnet.NOUN


def pos_tagger(text: str):
    tokens = word_tokenize(text)
    return pos_tag(tokens, tagset="universal")


def lemmatize(tagged_tokens: list[tuple[str, str]]):
    lemmatizer = WordNetLemmatizer()
    result = []
    for term, pos in tagged_tokens:
        wn_pos = _to_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(term, wn_pos)
        result.append((lemma, pos))
    return result


def main() -> None:
    tagged = pos_tagger("The quick brown foxes are jumping over the lazy dogs.")
    print("POS:", tagged)
    lemmas = lemmatize(tagged)
    print("LEM:", lemmas)


if __name__ == "__main__":
    main()
