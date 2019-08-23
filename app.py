from fastai.collab import *
from flask import Flask, request
import requests
import os.path


path = ''

export_file_url = 'https://www.dropbox.com/s/vj7cl45po1t86o8/cf1_books.pkl?dl=1'
export_file_name = 'cf1_books.pkl'


def down_load_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
            
def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        down_load_file(filename, url)
        return True
    return False

download_if_not_exists(export_file_name, export_file_url)

learn = load_learner(path, export_file_name)

books = pd.read_csv('books.csv')

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        
        user_id = int(request.form.get('user_id'))
        
        book_id_1 = int(request.form.get('book_id_1'))
        book_id_2 = int(request.form.get('book_id_2'))
        book_id_3 = int(request.form.get('book_id_3'))
        book_id_4 = int(request.form.get('book_id_4'))
        book_id_5 = int(request.form.get('book_id_5'))
        
        
        
      
        
        inference_df = pd.DataFrame(columns=['user_id', 'book_id'])
        inference_df.loc[0] = [user_id, book_id_1]
        inference_df.loc[1] = [user_id, book_id_2]
        inference_df.loc[2] = [user_id, book_id_3]
        inference_df.loc[3] = [user_id, book_id_4]
        inference_df.loc[4] = [user_id, book_id_5]
        
        
        
        
        inf_row_1 = inference_df.iloc[0]
        inf_row_2 = inference_df.iloc[1]
        inf_row_3 = inference_df.iloc[2]
        inf_row_4 = inference_df.iloc[3]
        inf_row_5 = inference_df.iloc[4]
        
        pred_1 = learn.predict(inf_row_1)
        pred_2 = learn.predict(inf_row_2)
        pred_3 = learn.predict(inf_row_3)
        pred_4 = learn.predict(inf_row_4)
        pred_5 = learn.predict(inf_row_5)
        
        rating_pred_1= float('%.2f'%(pred_1[0].data[0]))
        rating_pred_2= float('%.2f'%(pred_2[0].data[0]))
        rating_pred_3= float('%.2f'%(pred_3[0].data[0]))
        rating_pred_4= float('%.2f'%(pred_4[0].data[0]))
        rating_pred_5= float('%.2f'%(pred_5[0].data[0]))
        
        
        book_name_1 = books[books['id']==book_id_1]['title'].iloc[0]
        book_name_2 = books[books['id']==book_id_2]['title'].iloc[0]
        book_name_3 = books[books['id']==book_id_3]['title'].iloc[0]
        book_name_4 = books[books['id']==book_id_4]['title'].iloc[0]
        book_name_5 = books[books['id']==book_id_5]['title'].iloc[0]
        
        rec_df = pd.DataFrame(columns=['rating_pred', 'book_name'])
        rec_df.loc[0] = [rating_pred_1, book_name_1]
        rec_df.loc[1] = [rating_pred_2, book_name_2]
        rec_df.loc[2] = [rating_pred_3, book_name_3]
        rec_df.loc[3] = [rating_pred_4, book_name_4]
        rec_df.loc[4] = [rating_pred_5, book_name_5]
        
        final_rec = rec_df.loc[rec_df['rating_pred'].idxmax()]['book_name']
    
        
        return '''<h3>The input Consumer ID is: {}</h3>
                    The first input Book ID is: {}<br>
                    The name of the first book is: {}<br>
                    <b>The predicted rating the user will give on a sale of 1 to 5 for the first book is: {}</b><br>
                    The second input Book ID is: {}<br>
                    The name of the second book is: {}<br>
                    <b>The predicted rating the user will give on a sale of 1 to 5 for the second book is: {}</b><br>
                    The third input Book ID is: {}<br>
                    The name of the third book is: {}</br>
                    <b>The predicted rating the user will give on a sale of 1 to 5 for the third book is: {}</b><br>
                    The fourth input Book ID is: {}<br>
                    The name of the fourth book is: {}<br>
                    <b>The predicted rating the user will give on a sale of 1 to 5 for the fourth book is: {}</b><br>
                    The fifth input Book ID is: {}<br>
                    The name of the fifth book is: {}<br>
                    <b>The predicted rating the user will give on a sale of 1 to 5 for the fifth book is: {}</b><br>
                    <h1>It is recommended to offer the Consumer: {}</h1>
                    '''.format(user_id, book_id_1, book_name_1, rating_pred_1,
                              book_id_2, book_name_2, rating_pred_2,
                              book_id_3, book_name_3, rating_pred_3,
                              book_id_4, book_name_4, rating_pred_4,
                              book_id_5, book_name_5, rating_pred_5,
                              final_rec)


    return '''<form method="POST">
                  <h1>Predicting which book to offer an opted in consumer</h1>
                  
                  Enter a consumer ID between 1 and 53424: <input type="number" name="user_id" step=1 min=1 max =53424 required="required"><br>
                  
                  Input a Book ID number between 1 and 10000 that you might want to offer the Consumer: <input type="number" name="book_id_1" step=1 min=0 max =10000 required="required"><br>
                  Provide another Book ID number between 1 and 10000: <input type="number" name="book_id_2" step=1 min=0 max =10000 required="required"><br>
                  Provide another Book ID number between 1 and 10000: <input type="number" name="book_id_3" step=1 min=0 max =10000 required="required"><br>
                  Provide another Book ID number between 1 and 10000: <input type="number" name="book_id_4" step=1 min=0 max =10000 required="required"><br>
                  Provide another Book ID number between 1 and 10000: <input type="number" name="book_id_5" step=1 min=0 max =10000 required="required"><br>
                  
                  <input type="submit" value="Submit"><br>
              </form>'''

