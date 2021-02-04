# request, Blueprint,
from sanic import Blueprint, response
import os
# env = Environment(loader=PackageLoader('SpeechProcessing', 'templates'))

# from flask_socketio import SocketIO
REQUEST_API = Blueprint('request_api', __name__)

from config import configuration as config

from socketProcessing import SocketIOInput
from src.documentReaderQA import documentReaderQA
from src.transformer import transformerChat
from src.QAModel.transformerTrain import transformerTrain

trainModel = transformerTrain()
docReaderQA = documentReaderQA()
transformerModel = transformerChat()

transformerModel.buildModel()

def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


socketio = SocketIOInput()
def get_sockets():
    return socketio.blueprint()



@REQUEST_API.route('/', methods=['GET'])
async def root(request):
    # template = env.get_template('VUE.html')
    # html_content = template.render()
    # return html(html_content)
    return 'Hello QA Bot'
    # return await response.file('templates/VUE.html')

# @REQUEST_API.route('/js/<path:path>', methods=['GET'])
# async def send_js(request, path):
#     return await response.file('templates/js/'+ path)

## stella webhook
@REQUEST_API.route('/DocumentReader', methods=['POST'])
async def documentReaderWebhook(request):
    if request.method == 'POST':
        # check if the post request has the file part
        # return response.text(docReaderQA.predictAnswer(request.json))
        return response.text(docReaderQA.predictWikiAnswers(request.json['question']))

## Custom Document Reader webhook
@REQUEST_API.route('/CustomDocumentReader', methods=['POST'])
async def customDocumentReaderWebhook(request):
    if request.method == 'POST':
        # check if the post request has the file part
        # return response.text(docReaderQA.predictAnswer(request.json))
        return response.text(docReaderQA.predictWikiAnswers(request.json['question']))

## stella botframework webhook
@REQUEST_API.route('/TransformerBot', methods=['POST'])
async def transformerBotWebhook(request):
    if request.method == 'POST':
        # check if the post request has the file part
        # await botframeworkAPI.sendMessageToBot(request.json)
        res = await transformerModel.predict(request.json)
        return response.json(res)

# ## stella train transformer model
# @REQUEST_API.route('/TransformerBot/train', methods=['POST'])
# async def trainTransformerModel(request):
#     if request.method == 'POST':
#         # check if the post request has the file part
#         return response.json(trainModel.datasetPreprocess_train())

# ## stella train transformer model
# @REQUEST_API.route('/TransformerBot/modelList', methods=['POST'])
# async def transformerModelList(request):
#     if request.method == 'POST':
#         # check if the post request has the file part
#         return response.json(transformerModel.modelList())


## stella change transformer model
# @REQUEST_API.route('/TransformerBot/changeModel', methods=['POST'])
# async def changeTransformerModel(request):
#     if request.method == 'POST':
#         modelTime = request.json.get("timeStamp", None)
#         if modelTime == None or not os.path.isdir("models/"+modelTime):
#             response.json({"message":"No model is present to activate", "type":"s"})
#         config.modelPath = f"models/{modelTime}/my_model"
#         config.tokenizerPath = f"models/{modelTime}/tokenizer.pickle"
#         transformerModel.buildModel()
#         # check if the post request has the file part
#         return response.json({"message":f"{modelTime} model is activated", "type":"s"})


##d azTrDtransformer model
@REQUEST_API.route('/TransformerBot/Model', methods=['GET', 'POST'])
@REQUEST_API.route('/TransformerBot/Model/<id>', methods=["PUT", "DELETE"])
async def changeTransformerModel(request, id=None):
    if request.method == 'GET':
        modelList = await transformerModel.modelList()
        res = await trainModel.filterRunningModel(modelList)
        return response.json(res)
    if request.method == 'POST':
        train =  await trainModel.safeLoaderTimerTrain()
        return response.json(train)
    if request.method == 'PUT':
        modelTime = id
        if modelTime == None or not os.path.isdir("models/"+modelTime):
            return response.json({"message":"No model is present to activate", "type":"s"})
        config.modelPath = f"models/{modelTime}/my_checkpoint"
        config.tokenizerPath = f"models/{modelTime}/tokenizer.pickle"
        transformerModel.buildModel()
        # check if the post request has the file part
        return response.json({"message":f"{modelTime} model is activated", "type":"s"})
    if request.method == 'DELETE':
        res = await transformerModel.deleteModel(id)
        return response.json(res)





# @REQUEST_API.errorhandler(InvalidUsage)
# def handle_invalid_usage(error):
#     response = jsonify(error.to_dict())
#     response.status_code = error.status_code
#     return response

