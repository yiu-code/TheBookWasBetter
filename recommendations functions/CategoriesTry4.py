import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.feature_extraction import text # for CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


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
    # Replace category values with categorie names
    print("Replacing category numbers to list of strings...")
    index = list(categories.index.values)
    i_index = 0
    for cat_list in categories:
        rep_val = []
        for cat in cat_list:
            rep_val.append(cat_dict['category_name'][cat])
        categories[index[i_index]] = rep_val
        i_index += 1
    print("Done")
    return categories


def replace_singe_book_categories_number_to_words(categories):
    # Omdat ik de volgorde van de code later gewijzigd heb, ben ik te lui om het mooi in 1 functie te zetten
    # Ik weet dat 2x hetzelfde inladen en praktisch hetzelfde met de data doen bad practise is.
    df = pd.read_csv("C:/Users/ginad/Documents/#HR_2020_2021/Minor_DataScience/TheBookWasBetter/dataset/categories.csv")
    cat_dict = df.set_index('category_id').to_dict()
    rep_val = []
    for cat in categories:
        rep_val.append(cat_dict['category_name'][cat])
    categories = rep_val
    return categories


def single_book(data):
    """
    BOEKEN OM MEE TE OEFENEN:
        50 Knitted Dolls
        Botanical Gazette (1897)
        Harry Potter and the Deathly Hallows
        Make a Pop Rocket
        Aztecs and Incas
        Grave Matters : A Demon Trappers Novella
        LEGO DC Comics Super Heroes: Sidekick Showdown!
        Stories of Witches
        Norse Myths and Legends
    """
    found = 0
    select = str(input("Enter a book title: "))
    print("Searching for title...")
    for lines in data["Title"]:
        if select in lines:
            categories_found = data["Categories"].loc[data["Title"] == select].values
            if len(categories_found) > 0:
                print("\n'{}' has been found in the database.\n".format(select))
                found += 1
                # return title and list of categories
                return(select, categories_found[0])
    if found == 0:
        print("'{}' has not been found in the database.".format(select))
        quit()


def filter_data_on_categories(data, title, categories):
    findData = []
    categories = alter_categories_format(categories)
    print("Searching for matching categories..")
    for cat in categories:
        for line in data.iterrows():
            linelist = alter_categories_format(line[1]["Categories"])
            if cat in linelist:
                findData.append(line[1])
    print("Done")
    filteredData = pd.DataFrame(findData, columns =['Categories', 'Title', 'Description']) 
    filteredData = filteredData[filteredData["Title"] != title]
    filteredData = filteredData.drop_duplicates()
    return(filteredData, categories)


def train_vectorizer(bookCategories):
    vectoriser = text.CountVectorizer().fit(bookCategories)
    return(vectoriser, vectoriser.vocabulary_)


def vector_cosine(vector, compareList):
    # Vectorise text and compute the cosine similarity
    query0 = vector.transform([" ".join(vector.get_feature_names())])
    query1 = vector.transform(compareList)
    cos = cosine_similarity(query0.A, query1.A)
    return(query1, np.round(cos.squeeze(), 3))


def analyse_categories(filData, vector):
    csList = []
    for lines in filData["Categories"]:
        line_vec, line_cos = vector_cosine(vector, [' '.join(lines)])
        csList.append(line_cos)
    filData["cs_Categories"] = csList
    return(filData)


def add_title_to_categories_list(filData):
    modified_data = []
    cattle_data = filData[["Title", "Categories"]].values.tolist()
    for combo in cattle_data:
        flat_list = combo[0].split(" ")
        for item in combo[1]:
            flat_list.append(item)
        modified_data.append(flat_list)
    filData["Categories&Title"] = modified_data
    return(filData)
    

def adapt_single_book_data(title, categories):
    single_book = title.split(" ")
    for category in categories:
        single_book.append(category)
    return(single_book)


def analyse_categories_title(filData, vector):
    csList = []
    for lines in filData["Categories&Title"]:
        line_vec, line_cos = vector_cosine(vector, [' '.join(lines)])
        csList.append(line_cos)
    filData["cs_Categories&Title"] = csList
    return(filData)


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
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    data = prep_data()
    indices = pd.Series(data.index, index=data['Title'])
    # OM TE KIJKEN HOE VAAK EEN SINGLE CATEGORIE VOORKOMT
    #catSelection = single_category(data)

    title, categories = single_book(data)
    filtering, categories = filter_data_on_categories(data, title, categories)
    categories = replace_singe_book_categories_number_to_words(categories)
    print("'{}' contains the following categories: {}.\n".format(title, categories))

    filtering["Categories"] = data["Categories"].apply(alter_categories_format)
    filtering["Categories"] = replace_categories_number_to_words(filtering["Categories"])

    train = train_vectorizer(categories)
    filterCategories = filtering[["Title", "Categories"]]
    analyse = analyse_categories(filterCategories, train[0])
    print("\n\nRecommended based on categories:")
    print(analyse.nlargest(10, "cs_Categories"))

    categories_and_title = add_title_to_categories_list(filtering)
    single_book_data = adapt_single_book_data(title, categories)
    train_cattle = train_vectorizer(single_book_data)
    filterCategoriesTitle = filtering[["Title", "Categories&Title"]]
    analyse_cattle = analyse_categories_title(filterCategoriesTitle, train[0])
    print("\n\nRecommended based on categories and title:")
    print(analyse_cattle.nlargest(10, "cs_Categories&Title"))
    

main()