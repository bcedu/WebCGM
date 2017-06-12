import os
from flask import Flask, request, render_template, send_from_directory, abort, json, make_response, Response, jsonify
from werkzeug.utils import secure_filename
from PredictCGM.CGMLoader import Loader
from datetime import datetime
import argparse


class CGMServer(object):

    # Prepare the server
    app = Flask(__name__)
    # Init vars
    loader = None
    predicter_name = None
    voc_size = None
    seq_len = None
    predicter_dict = {}

    def __init__(self, host, port, loader, predicter_name="SVMPredicter", voc_size=15, seq_len=3):
        # Create uploads foler
        if not os.path.exists('./uploads/'):
            os.mkdir('./uploads/')
        CGMServer.app.config['UPLOAD_FOLDER'] = './uploads/'

        # set loader
        CGMServer.loader = loader
        print("Conected to {0}".format(self.loader.db_connection))

        # predicter options
        CGMServer.predicter_name = predicter_name
        CGMServer.voc_size = voc_size
        CGMServer.seq_len = seq_len
        CGMServer.predicter_dict = {}

        # Start the server
        CGMServer.app.run(host=host, port=port)

    @staticmethod
    @app.route('/', methods=['GET'])
    def init():
        try:
            pacients = CGMServer.loader.get_all_pacients()
            date = datetime.now().strftime("%Y-%m-%d %H:%M")
            params = {'pacients': pacients, 'time': date}
            return render_template('home.html', params=params)
        except Exception:
            # return index.html if database can not be opened
            return render_template('index.html')

    @staticmethod
    @app.route('/file', methods=['POST'])
    def upload_file():
        try:
            if not request.files.get('arxiu', False):
                return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
            file = request.files['arxiu']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(CGMServer.app.config['UPLOAD_FOLDER'], filename))
                return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
        except Exception as e:
            return jsonify(message="Select a correct file."), 500

    @staticmethod
    @app.route('/session', methods=['POST'])
    def create_session():
        try:
            # Get and validate data of new session
            elems = request.json
            datetime.strptime(elems['date'], "%Y-%m-%d %H:%M")
            try: insuline = float(elems['insulin'])
            except: raise Exception("Wrong format of 'Insulin'.")
            assert insuline > 0, "Insulin must be greater than 0."
            try: carboh = float(elems['carboh'])
            except: raise Exception("Wrong format of 'Carbohydrates'.")
            assert carboh > 0, "Carbohydrates must be greater than 0."
            try: glucose = float(elems['glucose'])
            except: raise Exception("Wrong format of 'Glucose'.")
            assert glucose > 0, "Glucose must be greater than 0."
            data, hora = elems['date'].split(" ")
            data = "{0}/{1}/{2}".format(data.split("-")[2], data.split("-")[1], data.split("-")[0])
            # Create the id assigned to the new session
            id = "{0}_{1}_{2}".format(elems['pacient'], data, hora)
            # Create list with all the data of the new session
            session_info = [[
                id,
                elems['pacient'],
                elems['date'],
                carboh,
                insuline,
                0 if elems['exerciseBf'] == 'No' else 1,
                0 if elems['exerciseAf'] == 'No' else 1,
                0 if elems['alcohol'] == 'No' else 1,
                CGMServer.loader.get_tipus_apat(hora),
                glucose,
            ]]
            # Load info of other sessions from file
            if elems['fileToUpload']:
                filename = elems['fileToUpload']
                filename = filename.split("\\")[-1]
                CGMServer.loader.load_data_from_csv(filename)
                CGMServer.loader.generate_fdps(normalized=True)
            CGMServer.loader.commit()
            CGMServer.loader.add_new_sessions(session_info)
            CGMServer.loader.commit()
            return json.dumps({'success': True, 'data': id}), 200, {'ContentType': 'application/json'}
        except Exception as e:
            CGMServer.loader.rollback()
            return jsonify(message=str(e)), 500

    @staticmethod
    @app.route('/session/<string:id>', methods=['GET'])
    def get_session(id):
        pacient, data, hora = id.split("_")
        data = "{0}/{1}/{2}".format(data.split("-")[2], data.split("-")[1], data.split("-")[0])
        session_id = "{0}_{1}_{2}".format(pacient, data, hora)
        ant_sessions = CGMServer.loader.get_ant_sessions(session_id)
        image_div = CGMServer.loader.draw_sessions(ant_sessions)
        params = {'image': image_div, 'id': session_id}
        return render_template('session_view.html', params=params)

    @staticmethod
    @app.route('/predicter', methods=['POST'])
    def create_predicter():
        session_to_predict = request.form['session_id']
        if session_to_predict is None:
            return abort(500)
        from PredictCGM import CGMPredict as pred_module
        PredicterObj = getattr(pred_module, CGMServer.predicter_name)
        predicter = PredicterObj(loader=CGMServer.loader, seq_len=CGMServer.seq_len, words_num=CGMServer.voc_size)
        imatges = CGMServer.loader.draw_sequence(session_to_predict)
        all_words = CGMServer.loader.draw_vocabulary()
        types = CGMServer.loader.get_labels_of_sequence(session_to_predict)
        sequences_data = [(imatges[i], types[i]) for i in range(len(imatges))]
        CGMServer.predicter_dict.update({session_to_predict: predicter})
        params = {
            'sequences_data': sequences_data,
            'route': "/predicter/"+session_to_predict.replace("/", "+"),
            'vocabulary': all_words
        }
        return render_template('sequence_view.html', params=params)

    @staticmethod
    @app.route('/predicter/<string:id>', methods=['GET'])
    def show_result(id):
        session_to_predict = id.replace("+", "/")
        predicter = CGMServer.predicter_dict[session_to_predict]
        res_info = predicter.predict([session_to_predict])

        label = res_info[0]
        word = res_info[2][0][0]
        lconfidence = max(res_info[1][0])
        wconfidence = max(res_info[3][0][0])

        word_img = CGMServer.loader.draw_word(word, session_to_predict)

        all_words = CGMServer.loader.draw_vocabulary()

        seq = CGMServer.loader.draw_sequence(session_to_predict, names=False)
        types = CGMServer.loader.get_labels_of_sequence(session_to_predict)
        sequences_data = [(seq[i], types[i]) for i in range(len(seq))]
        word_img2 = CGMServer.loader.draw_word(word)
        sequences_data.append((word_img2, label))

        seqs_final = []
        similar_seq = predicter.get_similar_models(session_to_predict, 5)
        for s, c in similar_seq:
            words = CGMServer.loader.get_words_of_sequence(s)
            words = str(words).replace("'", "")[1:-1]
            seqs_final.append((words, c))
        params = {
            'res': word,
            'label': label,
            'lconfidence': lconfidence,
            'wconfidence': wconfidence,
            'result_img': word_img,
            'sequences_data':sequences_data,
            'vocabulary': all_words,
            'similar_seq': seqs_final
        }
        CGMServer.predicter_dict[session_to_predict] = None
        CGMServer.loader.delete_session(session_to_predict)
        return render_template('final_view.html', params=params)

    @staticmethod
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(CGMServer.app.config['UPLOAD_FOLDER'], filename)


def main(args):
    if args.new_db_from_file:
        # Create a new database using the information of given file
        print("Creating new database...")
        loader = Loader(db_name=args.database, new_db=True, resources_path="./uploads/")
        print("Loading data from {0}".format(args.new_db_from_file))
        loader.load_data_from_csv(args.new_db_from_file)
        loader.generate_fdps(normalized=True)
    else:
        # Try to open a database from "uploads" folder
        loader = Loader(db_name=args.database, new_db=False, resources_path="./uploads/")

    CGMServer(args.host, args.port, loader, args.predicter, args.vocabulay_size, args.sequences_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ht", "--host", default="127.0.0.1",
                        help="The hostname to listen on. Set this to '0.0.0.0' to have the server available externally as well. Defaults to '127.0.0.1'.")
    parser.add_argument("-p", "--port", default=5000, type=int,
                        help="Port where the server will run. By default it's 5000.")
    parser.add_argument("-nf", "--new_db_from_file",
                        help="Name of file with information to create the new database. It must be in uploads folder.")
    parser.add_argument("-db", "--database", default='cgm_database.db',
                        help="Database name. If '--new_db_from_file' is passed, it's the name of the new database. Otherwise it's the name of the existing database that will be used. This database must be in 'uploads' directory. By default it's 'cgm_database.db'.")
    parser.add_argument("-pd", "--predicter", default='SVMPredicter',
                        choices=['KNNPredicter', 'SVMPredicter', 'SVM_KNN_Predicter'],
                        help="Predicter name. It must be  'KNNPredicter', 'SVMPredicter' or 'SVM_KNN_Predicter'. By default it's 'SVMPredicter'.")
    parser.add_argument("-s", "--sequences_len", default=3, type=int,
                        help="Lenght of the sequences used in predictions. Default is 3.")
    parser.add_argument("-v", "--vocabulay_size", default=5, type=int,
                        help="Size of the vocabularies of glucose histograms used in predictions. Default is 5.")
    main(parser.parse_args())

