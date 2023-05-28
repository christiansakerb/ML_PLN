#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from Api_m09_model_deployment_2 import clasif_genre
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Genre Clasification API',
    description='Genre Clasification API')

ns = api.namespace('predict', 
     description='Clasificador de generos de peliculas')
   
parser = api.parser()

parser.add_argument(
    'Text', 
    type=str, 
    required=True, 
    help='Plot to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        arg1 = str(args['Text'])

        return {
         "result": clasif_genre(arg1)
         
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
