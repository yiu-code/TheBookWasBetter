import pandas as pd
from rake_nltk import Rake
import numpy as np
import numpy.core.defchararray as np_f # <-----
import re, string
import nltk
from nltk.tokenize import word_tokenize 
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def set_sw():
    stopw = set(stopwords.words('english'))
    extraWords= ['whence', 'here', 'show', 'were', 'why',"'s'","n't","'ve", 'n’t', 'the', 'whereupon', 'not', 'more', 'how', 'eight', 'indeed', 'i', 'only', 'via', 'nine', 're', 'themselves', 'almost', 'to', 'already', 'front', 'least', 'becomes', 'thereby', 'doing', 'her', 'together', 'be', 'often', 'then', 'quite', 'less', 'many', 'they', 'ourselves', 'take', 'its', 'yours', 'each', 'would', 'may', 'namely', 'do', 'whose', 'whether', 'side', 'both', 'what', 'between', 'toward', 'our', 'whereby', "'m", 'formerly', 'myself', 'had', 'really', 'call', 'keep', "'re", 'hereupon', 'can', 'their', 'eleven', '’m', 'even', 'around', 'twenty', 'mostly', 'did', 'at', 'an', 'seems', 'serious', 'against', "n't", 'except', 'has', 'five', 'he', 'last', '‘ve', 'because', 'we', 'himself', 'yet', 'something', 'somehow', '‘m', 'towards', 'his', 'six', 'anywhere', 'us', '‘d', 'thru', 'thus', 'which', 'everything', 'become', 'herein', 'one', 'in', 'although', 'sometime', 'give', 'cannot', 'besides', 'across', 'noone', 'ever', 'that', 'over', 'among', 'during', 'however', 'when', 'sometimes', 'still', 'seemed', 'get', "'ve", 'him', 'with', 'part', 'beyond', 'everyone', 'same', 'this', 'latterly', 'no', 'regarding', 'elsewhere', 'others', 'moreover', 'else', 'back', 'alone', 'somewhere', 'are', 'will', 'beforehand', 'ten', 'very', 'most', 'three', 'former', '’re', 'otherwise', 'several', 'also', 'whatever', 'am', 'becoming', 'beside', '’s', 'nothing', 'some', 'since', 'thence', 'anyway', 'out', 'up', 'well', 'it', 'various', 'four', 'top', '‘s', 'than', 'under', 'might', 'could', 'by', 'too', 'and', 'whom', '‘ll', 'say', 'therefore', "'s", 'other', 'throughout', 'became', 'your', 'put', 'per', "'ll", 'fifteen', 'must', 'before', 'whenever', 'anyone', 'without', 'does', 'was', 'where', 'thereafter', "'d", 'another', 'yourselves', 'n‘t', 'see', 'go', 'wherever', 'just', 'seeming', 'hence', 'full', 'whereafter', 'bottom', 'whole', 'own', 'empty', 'due', 'behind', 'while', 'onto', 'wherein', 'off', 'again', 'a', 'two', 'above', 'therein', 'sixty', 'those', 'whereas', 'using', 'latter', 'used', 'my', 'herself', 'hers', 'or', 'neither', 'forty', 'thereupon', 'now', 'after', 'yourself', 'whither', 'rather', 'once', 'from', 'until', 'anything', 'few', 'into', 'such', 'being', 'make', 'mine', 'please', 'along', 'hundred', 'should', 'below', 'third', 'unless', 'upon', 'perhaps', 'ours', 'but', 'never', 'whoever', 'fifty', 'any', 'all', 'nobody', 'there', 'have', 'anyhow', 'of', 'seem', 'down', 'is', 'every', '’ll', 'much', 'none', 'further', 'me', 'who', 'nevertheless', 'about', 'everywhere', 'name', 'enough', '’d', 'next', 'meanwhile', 'though', 'through', 'on', 'first', 'been', 'hereby', 'if', 'move', 'so', 'either', 'amongst', 'for', 'twelve', 'nor', 'she', 'always', 'these', 'as', '’ve', 'amount', '‘re', 'someone', 'afterwards', 'you', 'nowhere', 'itself', 'done', 'hereafter', 'within', 'made', 'ca', 'them', 'her', 'during', 'among', 'thereafter', 'only', 'hers', 'in', 'none', 'with', 'un', 'put', 'hence', 'each', 'would', 'have', 'to', 'itself', 'that', 'seeming', 'hereupon', 'someone', 'eight', 'she', 'forty', 'much', 'throughout', 'less', 'was', 'interest', 'elsewhere', 'already', 'whatever', 'or', 'seem', 'fire', 'however', 'keep', 'detail', 'both', 'yourselves', 'indeed', 'enough', 'too', 'us', 'wherein', 'himself', 'behind', 'everything', 'part', 'made', 'thereupon', 'for', 'nor', 'before', 'front', 'sincere', 'really', 'than', 'alone', 'doing', 'amongst', 'across', 'him', 'another', 'some', 'whoever', 'four', 'other', 'latterly', 'off', 'sometime', 'above', 'often', 'herein', 'am', 'whereby', 'although', 'who', 'should', 'amount', 'anyway', 'else', 'upon', 'this', 'when', 'we', 'few', 'anywhere', 'will', 'though', 'being', 'fill', 'used', 'full', 'thru', 'call', 'whereafter', 'various', 'has', 'same', 'former', 'whereas', 'what', 'had', 'mostly', 'onto', 'go', 'could', 'yourself', 'meanwhile', 'beyond', 'beside', 'ours', 'side', 'our', 'five', 'nobody', 'herself', 'is', 'ever', 'they', 'here', 'eleven', 'fifty', 'therefore', 'nothing', 'not', 'mill', 'without', 'whence', 'get', 'whither', 'then', 'no', 'own', 'many', 'anything', 'etc', 'make', 'from', 'against', 'ltd', 'next', 'afterwards', 'unless', 'while', 'thin', 'beforehand', 'by', 'amoungst', 'you', 'third', 'as', 'those', 'done', 'becoming', 'say', 'either', 'doesn', 'twenty', 'his', 'yet', 'latter', 'somehow', 'are', 'these', 'mine', 'under', 'take', 'whose', 'others', 'over', 'perhaps', 'thence', 'does', 'where', 'two', 'always', 'your', 'wherever', 'became', 'which', 'about', 'but', 'towards', 'still', 'rather', 'quite', 'whether', 'somewhere', 'might', 'do', 'bottom', 'until', 'km', 'yours', 'serious', 'find', 'please', 'hasnt', 'otherwise', 'six', 'toward', 'sometimes', 'of', 'fifteen', 'eg', 'just', 'a', 'me', 'describe', 'why', 'an', 'and', 'may', 'within', 'kg', 'con', 're', 'nevertheless', 'through', 'very', 'anyhow', 'down', 'nowhere', 'now', 'it', 'cant', 'de', 'move', 'hereby', 'how', 'found', 'whom', 'were', 'together', 'again', 'moreover', 'first', 'never', 'below', 'between', 'computer', 'ten', 'into', 'see', 'everywhere', 'there', 'neither', 'every', 'couldnt', 'up', 'several', 'the', 'i', 'becomes', 'don', 'ie', 'been', 'whereupon', 'seemed', 'most', 'noone', 'whole', 'must', 'cannot', 'per', 'my', 'thereby', 'so', 'he', 'name', 'co', 'its', 'everyone', 'if', 'become', 'thick', 'thus', 'regarding', 'didn', 'give', 'all', 'show', 'any', 'using', 'on', 'further', 'around', 'back', 'least', 'since', 'anyone', 'once', 'can', 'bill', 'hereafter', 'be', 'seems', 'their', 'myself', 'nine', 'also', 'system', 'at', 'more', 'out', 'twelve', 'therein', 'almost', 'except', 'last', 'did', 'something', 'besides', 'via', 'whenever', 'formerly', 'cry', 'one', 'hundred', 'sixty', 'after', 'well', 'them', 'namely', 'empty', 'three', 'even', 'along', 'because', 'ourselves', 'such', 'top', 'due', 'inc', 'themselves', 'ii', 'th', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'mr', 'us', 'dk', 'workthis', 'youve', 'ive', 'sheve', 'heve', 'weve', 'theyve', 'didnt', 'dont', 'wouldnt', 'couldnt', 'cant', 'shouldnt', 'gl','cm', '________________________________', 'wwwultimatewritercom', 'svd', 'psg', 'zfxpsg', 'editorairsoftpresscom', 'iihf', 'miif','cae']
    for word in extraWords:
        stopw.add(word)
    return stopw


def prep_data():
    # df is the Goodreads dataset from Kaggle. The dataset is cleaned based on the language of the data and the formats of the book (paperback,
    # hardcover and book). After filtering we'll need the categories, titles and descriptions to continue.
    # Last line: Fill empty description with unknown instead of removing them.
    formats = ["1","2","9"]
    df = pd.read_csv("C:/Users/ginad/Documents/#HR_2020_2021/Minor_DataScience/TheBookWasBetter/dataset/dataset.csv",encoding='utf-8')
    df = df[df.lang == "en"]
    df = df[df.format.isin(formats)]
    df = df.reset_index(drop=True)
    data = df[["categories","title", "description"]]
    data.columns = ["Categories","Title", "Description"]
    data['Description'].fillna("unknown", inplace=True)
    return data


def remove_noise(text, stop):
    # function to clean the description.
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.replace('\n', ' ')
    # Remove special characters
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    # Remove punctuation
    text = text.str.replace('[^\w\s]', '')
    # Remove numbers
    text = text.str.replace('\d+', '')
    # Remove Stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # Convert to string
    text = text.astype(str)
    return text


def combine_title_and_description(title, desc, stop):
    # Function to combine title with description, so title will become a keyword as well.
    combine = title + desc
    combine = remove_noise(combine, stop)
    return combine


def alter_categories_format(cat_list):
    # Alter string to list of integer numbers
    cat_list = cat_list.replace("[", "")
    cat_list = cat_list.replace("]", "")
    cat_list = cat_list.replace(" ", "")
    cat_list = cat_list.split(",")
    cat_list = list(map(int, cat_list))
    return cat_list


def replace_categories_number_to_words(categories):
    # Make a dictionary of categories.csv (cat_dict)
    df = pd.read_csv("C:/Users/ginad/Documents/#HR_2020_2021/Minor_DataScience/TheBookWasBetter/dataset/categories.csv")
    cat_dict = df.set_index('category_id').to_dict()
    # Replace categorie values with categorie names
    index = 0    
    for cat_list in categories:
        rep_val = []
        for cat in cat_list:
            rep_val.append(cat_dict['category_name'][cat])
        categories[index] = rep_val
        index += 1
    return categories


def single_book(data):
    """
    BOEKEN OM MEE TE OEFENEN:
        50 Knitted Dolls
        Botanical Gazette (1897)
        Harry Potter and the Deathly Hallows
    """
    found = 0
    select = str(input("Enter a book title: "))
    for lines in data["Title"]:
        if select in lines:
            categories_found = data["Categories"].loc[data["Title"] == select].values
            if len(categories_found) > 0:
                print("\n'{}' has been found in the database.\n".format(select))
                print("'{}' contains the following categories: {}.\n".format(select, categories_found[0]))
                found += 1
                # return title and list of categories
                return(select, categories_found[0])
    if found == 0:
        print("'{}' has not been found in the database.".format(select))
        quit()


def filter_data_on_categories(data, title, categories):
    findData = []
    finalData = []
    for cat in categories:
        findData.append(data.index[data["Categories"].apply(lambda x: cat in x)])
        #findData.append(data[data["Categories"].apply(lambda x: cat in x)])
    print(findData)
    for single_list in findData:
        for single_cat in single_list:
            finalData.append(single_cat)
    finalData = set(finalData)
    return(finalData)


def term_frequency_inverse_data_frequency(data):
    tf = text.TfidfVectorizer(analyzer='word',min_df=0)
    tfidf_matrix = tf.fit_transform(data["Categories"])
    cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine)
    return(cosine)


def getRecommendations(data, title, categories, cosine):
    scores = []
    titles = data["Title"]
    for single_cat in categories:
        scores = scores + list(enumerate(cosine[single_cat]))
        scores = scores + list(enumerate(cosine))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        books = [i[0] for i in scores]
    return titles.iloc[books][len(categories):]


def single_category(data):
    """
    CATEGORIES OM MEE TE OEFENEN / CHECKEN:
        Adult & Contemporary Romance
        Cartoons
        Classic Horror
        Desserts
        Hobby and Gaming Books
        World Music
    """
    select = str(input("Enter a category: "))
    count = 0
    for lines in data["Categories"]:
        if select in lines:
            count += 1
    print("Category '{}' komt {} keer voor in de lijst".format(select, count))


def main():
    sw = set_sw()
    data = prep_data()
    indices = pd.Series(data.index, index=data['Title'])
    # create new column and fill the column with cleaned description
    data["cleaned_desc"] = combine_title_and_description(data["Title"], data["Description"], sw)
    # alter the str format of the values within categories
    data["Categories"] = data["Categories"].apply(alter_categories_format)
    # Replace category numbers (id's) with strings
    #data["Categories"] = replace_categories_number_to_words(data["Categories"])
    
    # OM TE KIJKEN HOE VAAK EEN SINGLE CATEGORIE VOORKOMT
    #catSelection = single_category(data)

    selection = single_book(data)
    filtering = filter_data_on_categories(data, selection[0], selection[1])
    #terms_cosine = term_frequency_inverse_data_frequency(filtering)
    #recommended = getRecommendations(filtering, selection[0], selection[1], terms_cosine)
    #print(recommended.head(10))

    # COSINE WITH ALL THE DATA
    #terms_cosine = term_frequency_inverse_data_frequency(data)
    #recommended = getRecommendations(data, selection[0], filtering, terms_cosine)


    #print(data.head())
    #oefen = [data.loc[0, :], data.loc[1, :]]
    #print(oefen[0])
    #print(oefen[1]["Categories"])
    #print(data.loc[0, :])
    #print(data.loc[1, :])
    #print(data["Categories"].value_counts().head())
    


main()