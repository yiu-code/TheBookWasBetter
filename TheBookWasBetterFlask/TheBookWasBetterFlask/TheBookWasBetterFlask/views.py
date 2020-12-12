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
from  sqlalchemy.sql.expression import func
from sqlalchemy import text
from flask_cors import CORS
import yaml
import json

#with open('./TheBookWasBetterFlask/goodreadsBookSimilarity.json') as json_file:
#    data = json.load(json_file)

#for post_dict in data:
#    db.session.add(DescriptionSimilarity(**post_dict))
#    db.session.commit()


@app.route('/')
@app.route('/home')
def index():
    newest_books = Book.query.filter_by(publication_year = datetime.now().year)\
                       .order_by(Book.book_id.desc())\
                       .with_entities(Book.book_id,Book.title, Book.image_url)\
                       .limit(5)

    best_rated_books = Book.query.filter(Book.average_rating > 4.8)\
                                 .order_by(Book.average_rating.desc() , func.random())\
                                 .with_entities(Book.book_id,Book.title, Book.image_url)\
                                 .limit(15)

    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        newest_books = newest_books,
        best_rated_books = best_rated_books
    ) 

@app.route('/browse', methods=['GET'], defaults={"page": 1}) 
@app.route('/browse/<int:page>', methods=['GET'])
def book(page=1):
    per_page = 100
    books = Book.query.order_by(Book.book_id)\
                      .with_entities(Book.book_id,Book.title, Book.image_url)\
                      .paginate(page,per_page,error_out=False)

    return render_template(
        'bookoverviewpage.html',
        title='Book Overview Page',
        year=datetime.now().year,
        books = books
    )

@app.route("/book/<book_id>")
def bookdetail(book_id):
    book = Book.query.filter_by(book_id = book_id).first()
    sim_desc = DescriptionSimilarity.query.filter_by(book_id = book_id)\
                                    .order_by(DescriptionSimilarity.cosine.desc())\
                                    .with_entities(DescriptionSimilarity.similar_book)\
                                    .all()
    sim_desc = [r for r, in sim_desc] 

    sim_books = Book.query.filter(Book.book_id.in_(sim_desc))\
                          .with_entities(Book.book_id,Book.title, Book.image_url)\
                          .all()

    print(sim_books)
    return render_template(
        'bookpage.html',
        title='Book Detail Page',
        year=datetime.now().year,
        book = book,
        sim_books = sim_books,
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
