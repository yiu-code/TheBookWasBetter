from TheBookWasBetterFlask import db

class DescriptionSimilarity(db.Model):
    __tablename__ = "DescriptionSimilarities"
    book_id = db.Column(db.Integer, primary_key=True)
    similar_book = db.Column(db.Integer, primary_key=True)
    cosine = db.Column(db.Float())
    user_feed = db.Column(db.Integer)

    def __init__(self, book_id, similar_book, cosine, user_feed):
        self.book_id = book_id
        self.similar_book = similar_book
        self.cosine = cosine
        self.user_feed = user_feed

    def __repr__(self):
        return '%s/%s/%s/%s' % (self.book_id, self.similar_book, self.cosine, self.user_feed)