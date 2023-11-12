import os
import requests
from flask import Flask, render_template, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app, supports_credentials=True, resources=r"/*")


def get_location_info(api_key, address):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        # print(data['error_message'])
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            formatted_address = result.get('formatted_address', 'N/A')
            latitude = result['geometry']['location']['lat']
            longitude = result['geometry']['location']['lng']
            place_id = result.get('place_id', 'N/A')
            return formatted_address, latitude, longitude, place_id
        else:
            return {'error': data['error_message']}
    except requests.exceptions.RequestException as e:
        return {'err': f"Something went wrong\n{e}"}


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        API_KEY = "AIzaSyApEmDjjoyu33Nvn3_5tohIMxPl6du9YNQ"
        address_query = request.form['address']

        location_info = get_location_info(API_KEY, address_query)

        print(location_info)

        if location_info['error']:
            return render_template('result.html', error_message=f'{location_info["error"]}')
        else:
            return render_template("index.html", location_info)


    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)