from TheBookWasBetterFlask import db

class Author(db.Model):
    __tablename__ = "Authors"
    author_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())

    def __init__(self, author_id, name):
        self.author_id = author_id
        self.name = name

    def __repr__(self):
        return '%s/%s' % (self.author_id, self.name)

