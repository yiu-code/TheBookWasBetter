library(dplyr)
library(tidyr)
#discovering data
df = read.csv("../dataset/dataset.csv")
formats = read.csv("../dataset/formats.csv")
authors = read.csv("../dataset/authors.csv")
categories = read.csv("../dataset/categories.csv")
#how big is the dataset?
print(dim(df))

# what columns do we have ?
print(colnames(df))

# what is the language distribution
pie(table(df$lang))
#The pie chart shows majority books are English
#as for PoC we will focus mainly on english books, so:
df = df %>% filter(lang == "en")

#what is the size now? 
print(dim(df))

#What kind of formats do we have?
#print(formats[order(formats$format_id),])

#what is the majority format?
    ##select items with a format id that exist in formats.csv
book_with_format = df %>% filter(format %in% c(1:38))
pie(table(book_with_format$format))

#alright we want to focus mainly on books therefore format id 1,2 and 9
df = df %>% filter(format %in% c(1,2,9))
print(dim(df))

#what about categories?
df$categories = gsub("[[]|[]]", "", df$categories)
df$authors = gsub("[[]|[]]", "", df$authors)
table(trimws(unlist(strsplit(df$categories, split = ","))))
