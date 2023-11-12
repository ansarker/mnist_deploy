import sqlite3
from PIL import Image
import torch
from torchvision import transforms
from datetime import datetime
from model import TestNet
import argparse


class MnistDB:
    def __init__(self, db_name="mnist.db"):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "./checkpoints/mnist-fashion-100.pth"
        self.classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    def create_table(self):
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS image_results (
                                id INTEGER PRIMARY KEY,
                                filename TEXT NOT NULL,
                                predicted_class TEXT,
                                timestamp TEXT
                            )
                            ''')
        self.connection.commit()
    
    def predict_image(self, image_path):
        model = TestNet().to(self.device)
        model.load_state_dict(torch.load(self.model_path))

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            output = model(image)
        
        _, predicted_class = torch.max(output, 1)
        return self.classes[predicted_class.item()]
    
    def add_image(self, image_path):
        predicted_class = self.predict_image(image_path)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute('''
                            INSERT INTO image_results (filename, predicted_class, timestamp) 
                            VALUES (?, ?, ?)
                            ''', (image_path, predicted_class, current_time))
        self.connection.commit()
    
    def get_record(self, record_id):
        self.cursor.execute('SELECT * FROM image_results WHERE id = ?', (record_id))
        return self.cursor.fetchone()
    
    def get_all_records(self):
        self.cursor.execute('SELECT * FROM image_results')
        return self.cursor.fetchall()
    
    def update_record(self, record_id, predicted_class):
        self.cursor.execute('UPDATE image_results SET predicted_class = ? WHERE id = ?', (predicted_class, record_id))
        self.connection.commit()
    
    def delete_record(self, record_id):
        self.cursor.execute('DELETE FROM image_results WHERE id = ?', (record_id,))
        self.connection.commit()
    
    def close_connection(self):
        self.connection.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--create', action='store_true', help='Create')
    parser.add_argument('-r', '--read', action='store_true', help='Read')
    parser.add_argument('-u', '--update', action='store_true', help='Update')
    parser.add_argument('-d', '--delete', action='store_true', help='Delete')
    parser.add_argument('-a', '--all', action='store_true', help="Get all record")
    parser.add_argument('-f', '--filepath', help="File to image path")
    parser.add_argument('--id', type=int, default=None, help="Record ID ex: 1")
    parser.add_argument('--update_data', type=str, help="Data to update")
    opt = parser.parse_args()

    db = MnistDB()
    db.create_table()

    if opt.create:
        db.add_image(opt.filepath)

    if opt.read:
        record = db.get_record(opt.id)
        print('Retrieved record: ', record)
    
    if opt.all:
        all_records = db.get_all_records()
        for record in all_records:
            print(record)

    if opt.update:
        if opt.id is None:
            print('You must enter record ID!')
        else:
            db.update_record(opt.id, opt.update_data)
            print(f'Record updated of id {opt.id}')

    if opt.delete:
        db.delete_record(opt.id)
        print(f'Deleted record {opt.id}')

    db.close_connection()

if __name__ == "__main__":
    main()