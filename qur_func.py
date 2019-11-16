import pandas as pd

abjad = {u"\u0627":'A',
u"\u0628":'b', u"\u062A":'t', u"\u062B":'v', u"\u062C":'j',
u"\u062D":'H', u"\u062E":'x', u"\u062F":'d', u"\u0630":'*', u"\u0631":'r',
u"\u0632":'z', u"\u0633":'s', u"\u0634":'$', u"\u0635":'S', u"\u0636":'D',
u"\u0637":'T', u"\u0638":'Z', u"\u0639":'E', u"\u063A":'g', u"\u0641":'f',
u"\u0642":'q', u"\u0643":'k', u"\u0644":'l', u"\u0645":'m', u"\u0646":'n',
u"\u0647":'h', u"\u0648":'w', u"\u0649":'Y', u"\u064A":'y'}


abjad[' ']=' '
abjad[u"\u0621"] = '\''
abjad[u"\u0623"] = '>'
abjad[u"\u0625"] = '<'
abjad[u"\u0624"] = '&'
abjad[u"\u0626"] = '}'
#abjad[u"\u0655"] = '\'' # Hamza below
abjad[u"\u0622"] = '|'
abjad[u"\u064E"] = 'a'
abjad[u"\u064F"] = 'u'
abjad[u"\u0650"] = 'i'
abjad[u"\u0651"] = '~'
abjad[u"\u0652"] = 'o'
abjad[u"\u064B"] = 'F'
abjad[u"\u064C"] = 'N'
abjad[u"\u064D"] = 'K'
abjad[u"\u0640"] = '_'
abjad[u"\u0670"] = '`'
abjad[u"\u0629"] = 'p'
abjad[u"\u0653"] = '^'
abjad[u"\u0654"] = '#'
abjad[u"\u0671"] = '{'
abjad[u"\u06DC"] = ':'
abjad[u"\u06DF"] = '@'
abjad[u"\u0653"] = '^'
abjad[u"\u06E0"] = '"'
abjad[u"\u06E2"] = '['
abjad[u"\u06E3"] = ';'
abjad[u"\u06E5"] = ','
abjad[u"\u06E6"] = '.'
abjad[u"\u06E8"] = '!'
abjad[u"\u06EA"] = '-'
abjad[u"\u06EB"] = '+'
abjad[u"\u06EC"] = '%'
abjad[u"\u06ED"] = ']'


# Create the reverse
alphabet = {}
for key in abjad:
    alphabet[abjad[key]] = key

def arabic_to_buc(ara):
    return ''.join(map(lambda x:abjad[x], list(ara)))


def buck_to_arabic(buc):
    if type(buc) is not str:
        return ''
    else:
        return ''.join(map(lambda x:alphabet[x], list(buc)))


def contains_num(x):
    import re 
    if type(x) is not str:
        return False
    else:
        found = re.search('\d', x)
        if found is None:
            return False
        else:
            return True


# function to return words given a list of sura
def sura_words(quran, s_list, kind='W'):
    if (kind=='R'):
        result = quran[quran.sura.isin(s_list)].Root.dropna().unique().tolist()
    elif (kind=='L'):
        result = quran[quran.sura.isin(s_list)].Lemma.dropna().unique().tolist()
    else:
        result = quran[quran.sura.isin(s_list)].FORM.unique().tolist()
    return [buck_to_arabic(x) for x in result]


# function to return words given a list of sura
def unique_sura_words(quran, s_list, kind='W'):
    if (kind=='R'):
        first = quran[quran.sura.isin(s_list)].Root.dropna().unique().tolist()
        second = quran[~quran.sura.isin(s_list)].Root.dropna().unique().tolist()
        result = list(set(first)-set(second))
    elif (kind=='L'):
        first = quran[quran.sura.isin(s_list)].Lemma.dropna().unique().tolist()
        second = quran[~quran.sura.isin(s_list)].Lemma.dropna().unique().tolist()
        result = list(set(first)-set(second))
    else:
        first = quran[quran.sura.isin(s_list)].FORM.dropna().unique().tolist()
        second = quran[~quran.sura.isin(s_list)].FORM.dropna().unique().tolist()
        result = list(set(first)-set(second))
    return [buck_to_arabic(x) for x in result]

#############################