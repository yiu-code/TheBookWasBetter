from TheBookWasBetterFlask import db

class Book(db.Model):
    __tablename__ = "Books"
    isbn = db.Column(db.String())
    asin = db.Column(db.String())
    average_rating = db.Column(db.Float())
    description = db.Column(db.String())
    authors = db.Column(db.ARRAY(db.JSON()))
    isbn13 = db.Column(db.String())
    publication_year = db.Column(db.Integer())
    image_url = db.Column(db.String())
    book_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String())
    categories =  db.Column(db.ARRAY(db.String()))
    
    def __init__(self, isbn, asin, average_rating, description, authors, isbn13, publication_year,image_url, book_id, title, categories):
        self.isbn = str(isbn)
        self.asin = str(asin)
        self.average_rating = average_rating
        self.description = description
        self.authors = authors
        self.isbn13 = str(isbn13)
        try:
            self.publication_year = int(float(publication_year))
        except:
            self.publication_year = None
        self.image_url = str(image_url)
        self.book_id = int(book_id)
        self.title = title
        self.categories = categories
    
    def __repr__(self):
        return '%s/%s/%s%s/%s/%s%s/%s/%s%s/%s' % (self.isbn, self.asin, self.average_rating, self.description, self.authors, self.isbn13, self.publication_year, self.image_url, self.book_id, self.title, self.categories)
