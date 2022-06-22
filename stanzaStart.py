import stanza


# Displacy docs
# https://github.com/explosion/spaCy/blob/master/spacy/displacy/render.py
# Serving display https://github.com/explosion/spaCy/blob/master/spacy/displacy/__init__.py
# https://spacy.io/api/token  <-- the doc object is a collection of tokens so go to the
# attributes part and match with what we have.
# https://spacy.io/api/doc    <-- go to attributes here, as well.



def first_testing():
    # this will also automatically download the models
    pipe = stanza.Pipeline("en")
    doc = pipe("This teacher is the best one I have ever had.")
    # print(doc)
    for word in doc.sentences[0].words:
        print(word)


def main():
    first_testing()


main()
