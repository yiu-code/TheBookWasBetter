"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from TheBookWasBetterFlask import app, db
from .models.book import Book
from .models.author import Author
from .models.descriptionSimilarity import DescriptionSimilarity
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_cors import CORS
import yaml
import json

#with open('./TheBookWasBetterFlask/goodreadsBookSimilarity.json') as json_file:
#    data = json.load(json_file)

#for post_dict in data:
#    db.session.add(DescriptionSimilarity(**post_dict))
#    db.session.commit()


#@app.route('/')
#@app.route('/home')
@app.route('/', methods=['GET'], defaults={"page": 1}) 
@app.route('/<int:page>', methods=['GET'])
def index(page=1):
    per_page = 100
    books = Book.query.order_by(Book.book_id).with_entities(Book.book_id,Book.title, Book.image_url).paginate(page,per_page,error_out=False)

    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        books = books
    )

#@app.route('/book/', methods=['GET'], defaults={book_id: "3"})
@app.route("/book/<book_id>")
def bookdetail(book_id):
    book = Book.query.filter_by(book_id = book_id).first_or_404()
    simDesc = DescriptionSimilarity.query.filter_by(book_id = book_id).order_by(DescriptionSimilarity.cosine.desc()).with_entities(DescriptionSimilarity.similar_book).limit(20)

    return render_template(
        'bookpage.html',
        title='Home Page',
        year=datetime.now().year,
        book = book,
        simDesc = simDesc
    )

#@app.route('/contact')
#def contact():
#    """Renders the contact page."""
#    return render_template(
#        'contact.html',
#        title='Contact',
#        year=datetime.now().year,
#        message='Your contact page.'
#    )

#@app.route('/about')
#def about():
#    """Renders the about page."""
#    return render_template(
#        'about.html',
#        title='About',
#        year=datetime.now().year,
#        message='Your application description page.'
#    )
