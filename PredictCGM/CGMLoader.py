import csv
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import sqlite3
import os
from tempfile import NamedTemporaryFile
import plotly
import plotly.graph_objs as go
import lmfit.models as mdl
from sklearn import preprocessing


class Loader(object):
    """
        This class is used to extract the data about the sessions of the patients from a .csv file and store it a database.

        It has all the methods related with data extraction, normalization and plotting.

        Provides the interface of the method:
            * generate_models
            * get_models
    """

    def __init__(self, db_name='default_cgm_database.db', new_db=False, resources_path="./"):
        """
        Creates or connect to a database.

        :param db_name: name of the new database or of the database used if 'new_db' is False
        :param new_db: if True, create a new database instead of connecting to the one with 'db_name'
        :param resources_path: path where the files will be placed and searched
        """
        self.min_time = 90
        self.max_time = 390

        self.resources_path = resources_path
        self.path = self.resources_path + db_name
        if not os.path.exists(self.resources_path):
            raise Exception("Resources directory doesn't exist: {0}".format(self.resources_path))
        # Time between the lectures of glucose will be taken
        self.db_connection = None
        self.cursor = None

        # Round all lectues to intervals of round_val
        self.round_val = 0.5

        if new_db:
            self._create_db_data()
        else:
            self._connect_database()

        # Default generator
        self.mgen = SequenceModels(WordsModels(SimpleModels(loader=self)))

    def _connect_database(self):
        """Set the cursor and open a db_connection."""
        self.db_connection = sqlite3.connect(self.path)
        self.cursor = self.db_connection.cursor()

    def commit(self):
        self.db_connection.commit()

    def rollback(self):
        self.db_connection.rollback()

    def _create_db_data(self):
        """Creates a new database and its tables. If a database existed, it is removed before creating the new one."""
        try:
            os.remove(self.path)
            print("Base de dades previa borrada")
        except: pass
        # Create new DB
        self.db_connection = sqlite3.connect(self.path)
        self.cursor = self.db_connection.cursor()
        self._create_users_table()
        self._create_sessions_table()
        self._create_lectures_cgm_table()
        self._create_words_table()
        self._create_seq_table()

    def _create_users_table(self):
        """Creates the table of Users."""
        self.cursor.execute(
            "CREATE TABLE Pacients "
            "(id TEXT PRIMARY KEY, edat INTEGER, sexe INTEGER, lower_target FLOAT, upper_target FLOAT);"
        )

    def _create_sessions_table(self):
        """Creates the table of Sessions."""
        self.cursor.execute(
            "CREATE TABLE Sessions "
            "(id TEXT PRIMARY KEY, pacient_id TEXT,  date TEXT, carboh FLOAT , insulin FLOAT , exerciseBf INTEGER, "
            "exerciseAf INTEGER, alcohol INTEGER, tipus_apat INTEGER, glucose FLOAT, word_id TEXT, label INTEGER, "
            "FOREIGN KEY(pacient_id) REFERENCES Pacients(id));"
        )

    def _create_lectures_cgm_table(self):
        """Creates the table of LecturesCGM."""
        self.cursor.execute(
            "CREATE TABLE LecturesCGM "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, sessio_id TEXT, lectura FLOAT, mtimestamp INTEGER , "
            "FOREIGN KEY(sessio_id) REFERENCES Sessions(id));"
        )
        self.cursor.execute(
            "CREATE TABLE CleanLecturesCGM "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, sessio_id TEXT, lectura FLOAT, "
            "FOREIGN KEY(sessio_id) REFERENCES Sessions(id));"
        )

    def _create_words_table(self):
        """Creates the table of Words."""
        self.cursor.execute("CREATE TABLE Words (id TEXT PRIMARY KEY, label INTEGER);")

    def _create_seq_table(self):
        """Creates the table of Sequences."""
        self.cursor.execute("CREATE TABLE Sequences (sessio_id TEXT PRIMARY KEY);")

    def load_data_from_csv(self, filename):
        """
        Load the data from 'filename' about patients and glucose lectures, 
        make a preprocess and store it in the database
        """
        self._generate_cgm_and_labels(filename)
        self.clean_data()

    def _generate_cgm_and_labels(self, filename):
        """
        Adds new patiens, sessions and lecures from 'filename' to tables 'Pacients', 'Sessions' and 'LecturesCGM'.
        Sessions without lectures are dissmissed.
        Each row is the information about a meal.
        Structure of data in file is expected as:
            
                patient_id; sex (1/0); lowertarget; middletarget; uppertarget; age; meal date; meal hour; timestamp; 
                    glucose; carbihidrats; insulin; exercice before (1/0); exercice after (1/0); alchool (1/0); lectures;
        
        where lectures have the format:
              val_lect1-date_lect1; val_lect2-date_lect2; ... ; val_lectN-date_lectN;
        """
        f = open('{0}/{1}'.format(self.resources_path, filename))
        reader = csv.reader(f, delimiter=';')
        for r in reader:
            use = False
            num = 0
            eid = ""
            gshape = []
            tshape = []
            res, min, max = 0, 0, 0
            time_hiper = 0
            session_info = []
            for elem in r:
                if num == 0:
                    eid = elem.replace(' ', '')
                elif num == 2:
                    min = float(elem)
                elif num == 4:
                    max = float(elem)
                elif num == 6:  # construim el id de la corva
                    aux = ''
                    for d in elem.split("/"):
                        if len(d) == 1:
                            d = '0{0}'.format(d)
                        aux += "{0}/".format(d)
                    eid = "{0}_{1}".format(eid, aux[:-1]).replace(' ', '')
                elif num == 7:
                    aux = ''
                    for d in elem.split(":"):
                        if len(d) == 1:
                            d = '0{0}'.format(d)
                        aux += "{0}:".format(d)
                    eid = "{0}_{1}".format(eid, aux[:-1]).replace(' ', '')
                    tipus_apat = self.get_tipus_apat(elem)
                elif num in [1, 5, 9, 10, 11, 12, 13, 14]:
                    session_info.append(float(elem))
                elif num > 14:
                    use = True
                    if len(elem.replace(',', '.').split('-')) != 2:
                        print(elem)
                    glevel, time_data = elem.replace(',', '.').split('-')
                    gdate = datetime.strptime(time_data, "%d/%m/%Y %H:%M:%S")
                    gshape.append(float(glevel))
                    tshape.append(gdate)

                    # Def. Hiper/Hipo 2
                    if gshape[-1] >= max:
                        time_hiper += 5
                    else:
                        time_hiper = 0
                    if gshape[-1] <= min:
                        res = -1
                    elif time_hiper >= 60 and res == 0:
                        res = 1

                    # Def. Hiper/Hipo 1
                    # if res == 0:
                    #     if gshape[-1] >= max:
                    #         res = 1
                    #     elif gshape[-1] <= min:
                    #         res = -1

                num += 1
            if use:
                session_info.append(tipus_apat)
                # Insert Pacient if it didn't exist
                try:
                    self.cursor.execute(
                        "INSERT INTO Pacients VALUES (?, ?, ?, ?, ?)",
                        (eid.split('_')[0], session_info[1], session_info[0], min, max)
                    )
                except: pass
                # Insert Sessions
                day = eid.split("_")[1]
                hour = eid.split("_")[2]
                date_session = "{0}-{1}-{2} {3}:00".format(
                    day.split("/")[2], day.split("/")[1], day.split("/")[0], hour
                )
                self.cursor.execute(
                    "INSERT INTO Sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (eid, eid.split('_')[0], date_session, session_info[3], session_info[4], session_info[5],
                     session_info[6], session_info[7], session_info[8], session_info[2], None, res)
                )
                # Insert Lect. values
                min = 5
                for value in gshape:
                    self.cursor.execute(
                        "INSERT INTO LecturesCGM (sessio_id, lectura, mtimestamp) VALUES (?, ?, ?)", (eid, value, min)
                    )
                    min += 5

    def get_tipus_apat(self, data_str):
        hora = int(data_str.split(":")[0])
        if hora in [3, 4, 5, 6, 7, 8, 9, 10]:
            return 1  # 'E'
        elif hora in [11, 12, 13, 14, 15, 16, 17]:
            return 2  # 'D'
        elif hora in [18, 19, 20, 21, 22, 23, 24, 0, 1, 2]:
            return 3  # 'S'

    def clean_data(self):
        """ Apply some methods to 'clean' and 'normalize' the glucose lectures stored in 'LecturesCGM':
                * filter data
                * Smooth
                * Standarize
                * Rescale
                * Round values to intervals of 0.025
            A new table 'CleanLecturesCGM' with the 'cleaned' lectures is created.
            Sessions without 'clean' lectures are deleted.
        """
        self._filter_cgm()
        self._smooth_cgm()
        # self._standarize_cgm()
        # self._rescale_cgm()
        self._round_lectures()

    def _filter_cgm(self):
        """
        Create table 'CleanLecturesCGM' with the lectures found between min_time and max_time of each session.
        The sessions without lectures between min_time and max_time are deleted.
        """
        cgm_ts = pd.read_sql(
            "SELECT sessio_id, lectura FROM LecturesCGM WHERE mtimestamp >= {0} AND mtimestamp < {1};".
                format(self.min_time, self.max_time),
            self.db_connection
        )
        cgm_ts.to_sql("CleanLecturesCGM", self.db_connection, index=False, if_exists="replace")
        sessions = list(self.cursor.execute("SELECT sessio_id FROM CleanLecturesCGM GROUP BY sessio_id;").fetchall())
        sessions = str(sessions)[1:-1].replace("(", "").replace("),", "").replace(",)", "")
        self.cursor.execute("DELETE FROM Sessions WHERE id NOT IN ({0})".format(sessions))

    def _round_lectures(self):
        """ Round all the values of lectures in CleanLecturesCGM to intervals of round_val """
        cgm_ts = pd.read_sql("SELECT sessio_id, lectura FROM CleanLecturesCGM;", self.db_connection)
        cgm_ts.lectura = cgm_ts.apply(lambda row: myround(row['lectura'], base=self.round_val), axis=1)
        cgm_ts.to_sql("CleanLecturesCGM", self.db_connection, index=False, if_exists="replace")
        self.min_lect = self.cursor.execute("SELECT min(lectura) FROM CleanLecturesCGM;").fetchone()[0]
        self.max_lect = self.cursor.execute("SELECT max(lectura) FROM CleanLecturesCGM;").fetchone()[0]

    def _rescale_cgm(self):
        """ 
        Rescaling of the data of CleanLecturesCGM from the original range so that all values are within 
        the range of 0 and 1.
        A value is rescaled as follows: (x - min) / (max - min)
        All the sessions are used in the rescale.
        """
        # load the dataset
        cgm_ts = pd.read_sql("SELECT sessio_id, lectura FROM CleanLecturesCGM;", self.db_connection)
        series = pd.Series(cgm_ts.lectura.values, index=cgm_ts.sessio_id.values)
        values = series.values
        values = values.reshape((len(values), 1))
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
        self.rescaler= scaler
        # normalize the dataset
        normalized = scaler.transform(values)
        cgm_ts['lectura'] = normalized
        cgm_ts.to_sql("CleanLecturesCGM", self.db_connection, index=False, if_exists="replace")

    def _standarize_cgm(self):
        """ 
        Standardizing the data of CleanLecturesCGM involves rescaling the distribution of values so that the mean of 
        observed values is 0 and the standard deviation is 1.
        A value is standardized as follows:
            y = (x - mean) / standard_deviation
            mean = sum(x) / count(x)
            standard_deviation = sqrt( sum( (x - mean)^2 ) / count(x))
        All the sessions are used.
        """
        # load the dataset
        cgm_ts = pd.read_sql("SELECT sessio_id, lectura FROM CleanLecturesCGM;", self.db_connection)
        series = pd.Series(cgm_ts.lectura.values, index=cgm_ts.sessio_id.values)
        values = series.values
        values = values.reshape((len(values), 1))
        # train the normalization
        scaler = StandardScaler()
        scaler = scaler.fit(values)
        self.standarizer = scaler
        # normalize the dataset
        standarized = scaler.transform(values)
        cgm_ts['lectura'] = standarized
        cgm_ts.to_sql("CleanLecturesCGM", self.db_connection, index=False, if_exists="replace")

    def _smooth_cgm(self):
        """ 
        Smooth the data of CleanLecturesCGM using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        """
        cgm_ts = pd.read_sql("SELECT sessio_id, lectura FROM CleanLecturesCGM;", self.db_connection)
        for sessio in self.cursor.execute("SELECT id FROM Sessions;"):
            serie = pd.read_sql(
                "SELECT * FROM CleanLecturesCGM WHERE sessio_id = '{0}'".format(sessio[0]), self.db_connection
            )
            lectures = serie.values[:, -1]
            w = 11
            if len(serie) < 12:
                w = len(serie)-1
                if w % 2 == 0:
                    w -= 1
            new = smooth(lectures, w)
            cgm_ts.ix[cgm_ts['sessio_id'] == sessio[0], -1] = new
        cgm_ts.to_sql("CleanLecturesCGM", self.db_connection, index=False, if_exists="replace")


    def generate_fdps(self, session_ids=None, normalized=False):
        """
        :param session_ids: sessions whose FDP will be generated. If none is given, the FDPs of all sessions in 
                            database will be generated
        :param normalized: if true, normalize the values of the fdp between 1 and 0
        """
        fdp_data = {}
        if session_ids is None:
            cgm_ts = pd.read_sql("SELECT sessio_id, lectura FROM CleanLecturesCGM;", self.db_connection)
            sessions = self.cursor.execute("SELECT id FROM Sessions WHERE label IS NOT NULL;").fetchall()
        else:
            session_ids = str(session_ids)[1:-1]
            cgm_ts = pd.read_sql(
                "SELECT sessio_id, lectura FROM CleanLecturesCGM WHERE sessio_id IN ({0});".format(session_ids),
                self.db_connection
            )
            sessions = self.cursor.execute("SELECT id FROM Sessions WHERE id IN ({0});".format(session_ids)).fetchall()

        values = [round(float(x), 5) for x in np.arange(self.min_lect, self.max_lect+self.round_val, self.round_val)]
        for session in sessions:
            h = cgm_ts[cgm_ts.sessio_id == session[0]].groupby(['sessio_id', 'lectura']).size()
            indexos = [round(float(x), 5) for x in h[session[0]].index]
            new_index = pd.Series({x: 0 for x in values if x not in indexos})
            fdp_values = h[session[0]].append(new_index).sort_index().tolist()
            if normalized:
                num = sum(fdp_values)
                fdp_values = [round(x / num, 5) for x in fdp_values]
            fdp_data[session[0]] = [session[0]] + fdp_values
        fdp_data = pd.DataFrame(list(fdp_data.values()), columns=['id'] + values)
        if session_ids is None:
            fdp_data.to_sql("SessionsFDP", self.db_connection, index=False, if_exists="replace")
        else:
            fdp_data.to_sql("SessionsFDP", self.db_connection, index=False, if_exists="append")

    def add_new_sessions(self, sessions_data):
        """
        :param sessions_data: list of tuples (one for session) with all the information about each session:
                * id
                * pacient_id
                * date
                * carboh
                * insulin
                * exerciseBf
                * exerciseAf
                * alcohol
                * tipus_apat
                * glucose
        Creates new sessions and adds it to the table 'Sessions'
        """
        session_ids = []
        for session_info in sessions_data:
            session_ids.append(session_info[0])
            self.cursor.execute(
                "INSERT INTO Sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_info[0], session_info[1], session_info[2], session_info[3], session_info[4], session_info[5],
                 session_info[6], session_info[7], session_info[8], session_info[9], None, None)
            )
        return session_ids

    def delete_session(self, session_id):
        try:
            self.cursor.execute("DELETE FROM Sessions WHERE id = '{0}'".format(session_id))
            self.cursor.execute("DELETE FROM SessionsNorm WHERE id = '{0}'".format(session_id))
            self.cursor.execute("DELETE FROM Sequences WHERE sessio_id = '{0}'".format(session_id)).fetchall()
            self.commit()
            print("deleted: {0}".format(session_id))
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def get_session_fields(name="SessionsNorm", weight=1):
        return "{0}.tipus_apat*{1}, {0}.insulin*{1}, {0}.glucose*{1}, {0}.carboh*{1}, {0}.alcohol*{1}, {0}.exerciseBf*{1} ".format(name, weight)

    def _get_all_sessions(self, pacient=None):
        """ Returns all session ids. If a patient is given return only the sessions of the patient."""
        if pacient is None:
            return [info[0] for info in self.cursor.execute("SELECT id FROM Sessions ORDER BY id;").fetchall()]
        else:
            return [info[0] for info in self.cursor.execute("SELECT id FROM Sessions WHERE pacient_id = '{0}' "
                                                            "ORDER BY id;".format(pacient)).fetchall()]

    def get_all_sessions(self, pacient=None, for_train=False):
        """ Returns all session ids. If a patient is given return only the sessions of the patient. """
        sessions = self._get_all_sessions(pacient)
        if for_train:
            try:
                train_sessions = [info[0] for info in
                                  self.cursor.execute("SELECT sessio_id FROM Sequences AS Sq JOIN Sessions AS S "
                                                      "ON Sq.sessio_id=S.id WHERE S.word_id IS NOT NULL ").fetchall()]
                sessions = [x for x in sessions if x in train_sessions]
            except:
                return sessions
        return sessions

    def get_all_pacients(self):
        """ Returns all patients ids"""
        return [info[0] for info in self.cursor.execute("SELECT id FROM Pacients;").fetchall()]

    def get_session_labels(self, session_ids):
        """ Returns the labels of given sessions """
        if not isinstance(session_ids, list):
            session_ids = [session_ids]
        return [info[0] for info in
                self.cursor.execute("SELECT label FROM Sessions WHERE id IN ({0}) "
                                    "ORDER BY id;".format(str(session_ids)[1:-1])).fetchall()]

    def get_ant_sessions(self, session_id, num=3):
        """ Return session ids of the 'num' previous sessions to 'session_id'"""
        pacient, date = self.cursor.execute(
            "SELECT pacient_id, date FROM Sessions WHERE id='{0}';".format(session_id)
        ).fetchone()
        return list(reversed([x[0] for x in
                              self.cursor.execute("SELECT id FROM Sessions WHERE pacient_id='{0}' AND date < '{1}' "
                                                  "ORDER BY date DESC LIMIT {2};".format(pacient, date, num)
                                                  ).fetchall()]))

    def get_labels_of_words(self, words_list):
        """ Return the labels of the words in words_list"""
        if not isinstance(words_list, list):
            words_list = [words_list]
        word_labels = dict([(x[0], x[1]) for x in self.cursor.execute(
            "SELECT id, label FROM Words WHERE id IN ({0}) ORDER BY id;".format(str(words_list)[1:-1])).fetchall()])
        labels = [word_labels[x] for x in words_list]
        return labels

    def get_labels_of_sequence(self, session_id):
        """ Return labels of sessions of sequence of session_id """
        seq_names = [tup[1] for tup in self.cursor.execute('PRAGMA TABLE_INFO(Sequences)').fetchall() if
                     tup[1] not in ['sessio_id', 'next_label']]
        seq_names = str(seq_names).replace("'", "")[1:-1]
        seq_info = self.cursor.execute(
            "SELECT {1} FROM Sequences WHERE sessio_id='{0}'".format(session_id, seq_names)
        ).fetchone()
        return self.get_session_labels([x for x in seq_info])

    def get_words_of_sequence(self, session_id):
        seq_names = [tup[1] for tup in self.cursor.execute('PRAGMA TABLE_INFO(Sequences)').fetchall() if
                     tup[1] not in ['sessio_id', 'next_label']]
        seq_names = str(seq_names).replace("'", "")[1:-1]
        seq_info = self.cursor.execute(
            "SELECT {1} FROM Sequences WHERE sessio_id='{0}'".format(session_id, seq_names)
        ).fetchone()
        return self.get_session_words([x for x in seq_info] + [session_id])

    def get_session_words(self, session_ids):
        """ Returns the word of given sessions """
        if not isinstance(session_ids, list):
            session_ids = [session_ids]
        return [info[0] for info in
                self.cursor.execute("SELECT word_id FROM Sessions WHERE id IN ({0}) ORDER BY id;".format(
                    str(session_ids)[1:-1])).fetchall()]

    def draw_sessions(self, session_ids):
        """ Draw all the lectures and clean_lectures of session ids in a single and continuous plot"""
        pacient = ''
        min = 0
        max = 0
        tdates = []
        tcgm = []
        filtered_plots = []
        y_marks = []
        x_marks = []
        for session_id in session_ids:
            cgm_filtered = pd.read_sql(
                "SELECT lectura FROM LecturesCGM WHERE sessio_id = '{0}' AND mtimestamp >= {1} AND mtimestamp < {2};"
                    .format(session_id, self.min_time, self.max_time),  self.db_connection
            )
            cgm = pd.read_sql(
                "SELECT lectura FROM LecturesCGM WHERE sessio_id = '{0}';".format(session_id), self.db_connection
            )
            pacient = session_id.split("_")[0]
            min, max = self.cursor.execute(
                "SELECT lower_target, upper_target FROM Pacients WHERE id = '{0}';".format(pacient)
            ).fetchone()
            ini_date = datetime.strptime(session_id.split("_")[1]+" "+session_id.split("_")[2], "%d/%m/%Y %H:%M")
            dates = []
            for i in range(len(cgm)):
                if len(dates) == 0:
                    dates.append(ini_date + timedelta(minutes=5))
                else:
                    dates.append(dates[-1] + timedelta(minutes=5))
            tdates = tdates + dates
            tcgm = tcgm + cgm.lectura.tolist()

            i = 0
            for i in range(len(dates)):
                if (dates[i] - ini_date).seconds >= (self.min_time * 60):
                    break
            j = 0
            for j in range(i, len(dates)):
                if (dates[j] - ini_date).seconds >= (self.max_time * 60):
                    break
            f_dates = dates[i:j]
            f_cgm = cgm_filtered.lectura.tolist()
            filtered_plots.append(
                go.Scatter(x=f_dates, y=f_cgm, mode='lines', name=session_id.replace("_", " "),
                           line=dict(color='#ff8700'))
            )

            y_marks.append(cgm.lectura.tolist()[0])
            x_marks.append(dates[0])

        # Draw
        layout = {
            'title': pacient,
            'shapes': [{
                'type': 'rect',
                'xref': 'paper',
                'yref': 'y',
                'x0': 0,
                'y0': min,
                'x1': 1,
                'y1': max,
                'fillcolor': '#b3ffb3',
                'opacity': 0.2,
                'line': {'width': 0}
            }],
        }

        fig = {
            'data': [go.Scatter(x=tdates, y=tcgm, name="Glucose", legendgroup="g1"),
                     go.Scatter(x=tdates, y=[max] * len(tdates), name="Upper Target", legendgroup="g3", mode='lines',
                                line=dict(color='#cc0000')),
                     go.Scatter(x=tdates, y=[min] * len(tdates), name="Lower Target", legendgroup="g3", mode='lines',
                                line=dict(color='#000000'))]
                    + filtered_plots
                    + [go.Scatter(x=x_marks, y=y_marks, name="Carbohydrates/Insulin", mode='markers', legendgroup="g4",
                                  line=dict(color='#40bf00'))],
            'layout': layout,
        }
        image = plotly.offline.plot(fig, auto_open=False, show_link=False, output_type='div')
        return image

    def draw_hist(self, session_id):
        """ Draw the histogram of lectures of a session"""
        cgm_ts = pd.read_sql(
            "SELECT sessio_id, lectura FROM LecturesCGM WHERE sessio_id = '{0}' AND mtimestamp >= {1}"
            " AND mtimestamp < {2};".format(session_id, self.min_time, self.max_time),  self.db_connection
        )
        cgm_ts = cgm_ts.lectura
        layout = {
            'title': "Used data Histogram",
        }
        fig = {
            'data': [go.Histogram(x=cgm_ts, autobinx=False, xbins=dict(start=2.05, end=20.05, size=0.1))],
            'layout': layout,
        }
        return plotly.offline.plot(fig, auto_open=False, show_link=False, output_type='div')

    def draw_word(self, word_id, session_id=None):
        """ Draw a word. If a session id is given, its used in the title of the plot """
        wcol_names = [tup[1] for tup in self.cursor.execute('PRAGMA TABLE_INFO(Words)').fetchall() if
                      tup[1] not in ['id', 'label']]

        wcol_names = str(wcol_names)[1:-1].replace("'", "")
        cols = wcol_names.replace("_", ".").replace("n", "")
        vals = self.cursor.execute("SELECT * FROM Words WHERE id = '{0}';".format(word_id)).fetchone()
        vals = vals[1:-1]
        name = word_id
        if session_id:
            name = "{0} / {1}".format(session_id, name)
        layout = {
            'title': name,
        }

        fig = {
            'data': [go.Bar(x=cols, y=vals, showlegend=False)],
            'layout': layout,
        }
        image = plotly.offline.plot(fig, auto_open=False, show_link=False, output_type='div')
        return image

    def draw_vocabulary(self):
        """ Draw all the words of the table 'Words' """
        imatges = []
        for word_id in self.cursor.execute("SELECT id FROM Words;").fetchall():
            imatges.append(self.draw_word(word_id[0]))
        return imatges

    def draw_sequence(self, session_id, names=True):
        """ Draw all sessions of sequence of session_id"""
        seq_names = [tup[1] for tup in self.cursor.execute('PRAGMA TABLE_INFO(Sequences)').fetchall() if tup[1] not in
                     ['sessio_id', 'next_label']]
        seq_names = str(seq_names)[1:-1].replace("'", "")
        sessions = self.cursor.execute(
            "SELECT {1} FROM Sequences WHERE sessio_id = '{0}';".format(session_id, seq_names)
        ).fetchone()
        images = []
        for id in sessions:
            word = self.cursor.execute("SELECT word_id FROM Sessions WHERE id = '{0}';".format(id)).fetchone()[0]
            if not names:
                id = None
            images.append(self.draw_word(word, id))
        return images

    def add_generator(self, generator_name):
        """
        Adds a new 'label' to the model generator by creating a new generator 'generator_name' using the current 
        generator.
        """
        self.mgen = generator_name(self.mgen)

    def set_generator(self, generator_obj):
        """ Replace current generator with generator_obj """
        self.mgen = generator_obj

    def generate_models(self, args_dict):
        """ Call 'generate_models' method of ModelGenerator"""
        self.mgen.generate_models(args_dict)

    def get_models(self, session_ids, args_dict=None):
        """ Call 'get_models' method of ModelGenerator"""
        return self.mgen.get_models(session_ids, args_dict)

    def get_models_label(self, session_ids):
        """ Call 'get_models_label' method of ModelGenerator"""
        return self.mgen.get_models_label(session_ids)

class ModelsGenerator(object):

    def __init__(self, model_generator=None, loader=None):
        if model_generator:
            self.loader = model_generator.loader
            self.child_mgen = model_generator
        elif loader:
            self.loader = loader
            self.child_mgen = None
        else:
            raise Exception("Loader required")
        self.cursor = self.loader.cursor

    def generate_models(self, args_dict):
        """ Generate models of sessions """
        if self.child_mgen:
            self.child_mgen.generate_models(args_dict)
        self._generate_models(args_dict)

    def get_models(self, session_ids, args_dict=None):
        """ Returns models of given sessions"""
        if args_dict is None:
            args_dict = {}
        return self._get_models(session_ids, args_dict)

    def get_models_label(self, session_ids):
        """ Returns labels of models of given sessions """
        return self._get_models_label(session_ids)

    def _generate_models(self, args_dict):
        return None

    def _get_models(self, session_ids, args_dict):
        return None

    def _get_models_label(self, session_ids):
        return self.loader.get_session_labels(session_ids)


class SimpleModels(ModelsGenerator):

    def _generate_models(self, args_dict):
        df = pd.read_sql("SELECT Sessions.* FROM Sessions", self.loader.db_connection)
        df.to_sql("SessionsNorm", self.loader.db_connection, if_exists="replace", index=False)

    def _get_models(self, session_ids, args_dict):
        return pd.read_sql("SELECT SessionsFDP.*, Sessions.label FROM Sessions INNER JOIN SessionsFDP "
                           "ON Sessions.id = SessionsFDP.id WHERE Sessions.id IN ({0});"
                           .format(str(session_ids)[1:-1]), self.loader.db_connection)

class NormalizedModels(SimpleModels):

    def _generate_models(self, args_dict):
        df = pd.read_sql("SELECT SessionsNorm.* FROM SessionsNorm", self.loader.db_connection)
        x = df.iloc[:, 3:-2].values
        min_max_scaler = preprocessing.MinMaxScaler()
        df_scaled = pd.DataFrame(min_max_scaler.fit_transform(x))
        final_df = df.iloc[:, :3]
        i = 0
        for col_name in df.iloc[0, 3:-2].index:
            final_df = final_df.assign(aux=df_scaled.iloc[:, i].values)
            final_df.columns = list(final_df.columns)[:-1] + [col_name]
            i += 1

        final_df = final_df.assign(aux=df.iloc[:, -2].values)
        final_df.columns = list(final_df.columns)[:-1] + ['word_id']

        final_df = final_df.assign(aux=df.iloc[:, -1].values)
        final_df.columns = list(final_df.columns)[:-1] + ['label']

        final_df.to_sql("SessionsNorm", self.loader.db_connection, if_exists="replace")

class WordsModels(ModelsGenerator):

    def _generate_models(self, args_dict):
        k = args_dict.get('k', 5)
        try:
            self.cursor.execute("DROP TABLE Words;")
        except:
            pass
        all_sessions = self.loader.get_all_sessions()
        models = self.child_mgen.get_models(all_sessions)
        for lb_value, lb_rep in [(lb, str(lb)) for lb in [-1, 0, 1]]:
            fdp_data = models.ix[models.label == lb_value]
            kmeans, words = self.create_vocabulary(k, fdp_data)
            new = NamedTemporaryFile(mode="w")
            new.write('id;')
            for index in fdp_data.columns[1:-1]:
                new.write("n{0};".format(index).replace(".", "_"))
            new.write('label\n')
            i = 0
            for word in words:
                i += 1
                new.write(("w{0}_{1};".format(lb_rep, i)+str(word[:].round(5).tolist())[1:-1]+"\n").replace(',', ';')
                          .replace("-0.0;", "0.0;").replace("0.0;", "0;"))
            labels = kmeans.predict(fdp_data[fdp_data.columns[1:]])
            for i in range(len(labels)):
                self.cursor.execute(
                    "UPDATE Sessions SET word_id='w{0}_{1}' WHERE id='{2}';"
                        .format(lb_rep, labels[i]+1, fdp_data.iloc[i, 0])
                )
            new.flush()
            new.seek(0)
            df = pd.DataFrame.from_csv(new.name, sep=';')
            df.to_sql("Words", self.loader.db_connection, if_exists="append")
            new.close()

    def _get_models(self, session_ids, args_dict):
        weight = args_dict.get("weight", 1)
        word_weight = args_dict.get("word_weight", 1)

        word_cols = str(["Words.{0}*{1}".format(tup[1], word_weight) for tup in
                         self.cursor.execute('PRAGMA TABLE_INFO(Words)').fetchall()
                         if tup[1] not in ['id', 'label']]).replace("'","")[1:-1]

        sstring = str(session_ids)[1:-1]
        pacients_str = "Pacients.sexe*{0}, ((1.0*Pacients.edat)/100)*{0}".format(weight)
        return pd.read_sql(
            "SELECT SessionsNorm.id, {3}, {0}, {1}, SessionsNorm.label FROM SessionsNorm INNER JOIN Sessions ON SessionsNorm.id = Sessions.id INNER JOIN Words ON Sessions.word_id = Words.id INNER"
            "  JOIN  Pacients ON Sessions.pacient_id=Pacients.id WHERE SessionsNorm.id IN ({2});"
                .format(self.loader.get_session_fields(weight=weight), word_cols, sstring, pacients_str), self.loader.db_connection
        )


    @staticmethod
    def create_vocabulary(k, cgm_models):
        """
        :param  k:  number of clusters for the kmeans algorithm
        :param  cgm_models: dataframe with the caracteristics of the models.
                            There is a row for each session with its respective model.
                            The dataframe has the folowing columns:
                              * unnamed: integer to identify each row
                              * id: string that identifies the session of the lecture: {PacientName}_{DD/MM/YYYY}_{h:m}.
                                The date and hour are the ones from the begining of the session.
                              * 1 column for each caracteristic of the model.
                              * label: says if its classified as hipo/hiper/normal
        :returns
        kmeans:     kmeans object made with the data of 'cgm_models'.
        centroids:  ndarray with the data about the centroids generated by 'kmeans'.
        """
        data = cgm_models[cgm_models.columns[1:]]
        if k > len(data):
            print("ERROR generating Vocabulary: words reduced from {0} to {1}".format(k, len(data)))
            k = len(data)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        centroids = kmeans.cluster_centers_
        for c in centroids:
            c[-1] = round(c[-1])
        return kmeans, centroids


class SequenceModels(ModelsGenerator):

    def _generate_models(self, args_dict):
        seq_len = args_dict.get('ns', 3)
        new_data = []
        for sessio_info in self.cursor.execute("SELECT id, date, pacient_id from Sessions;").fetchall():
            anteriors = self.cursor.execute(
                "SELECT id FROM Sessions WHERE pacient_id='{0}' AND date < '{1}' ORDER BY date DESC LIMIT {2};"
                    .format(sessio_info[2], sessio_info[1], seq_len)
            ).fetchall()
            aux_list = []
            for i in anteriors:
                aux_list.append(i[0])
            if len(aux_list) != seq_len:
                continue
            aux_list = [sessio_info[0]] + list(reversed(aux_list))
            new_data.append(aux_list)
        indexs = []
        for i in range(1, seq_len + 1):
            indexs.append("s{0}".format(i))
        new_df = pd.DataFrame(new_data, columns=['sessio_id'] + indexs)
        new_df.to_sql("Sequences", self.loader.db_connection, if_exists="replace", index=False)

    def _get_models(self, session_ids, args_dict):
        new_data = []
        used_models = {}
        ant_model = None
        weight = args_dict.get("weight", 1)
        sfields = self.loader.get_session_fields(weight=weight)
        pacients_str = "Pacients.sexe*{0}, ((1.0*Pacients.edat)/100)*{0}".format(weight)
        for session_id in session_ids:
            aux = [x for x in self.cursor.execute("SELECT SessionsNorm.id, {2}, {0} FROM SessionsNorm JOIN Pacients ON pacient_id = Pacients.id WHERE SessionsNorm.id = '{1}'"
                                                  .format(sfields, session_id, pacients_str)).fetchone()]
            ant_sessions = self.cursor.execute(
                "SELECT * FROM Sequences WHERE sessio_id = '{0}';".format(session_id)
            ).fetchone()
            if ant_sessions:
                ant_sessions = ant_sessions[1:]
                for ant_id in ant_sessions:
                    args_dict.update({"word_weight": self.calc_sequence_weight(session_id, ant_id)})
                    if used_models.get(ant_id, pd.DataFrame()).empty:
                        ant_model = self.child_mgen.get_models([ant_id], args_dict=args_dict)
                        used_models[ant_id] = ant_model
                    else:
                        ant_model = used_models.get(ant_id)
                    aux = aux + ant_model.iloc[:, 1:-1].values.tolist()[0]
                word = self.cursor.execute(
                    "SELECT word_id FROM Sessions WHERE id = '{0}';".format(session_id)
                ).fetchone()[0]
                aux.append(word)
                new_data.append(aux)
        names = ["id", "sexe", "edat"] + sfields.replace("SessionsNorm.", "").replace(" ", "").split(",")
        if ant_model is not None:
            aux = list(ant_model.columns.values)[1:-1]
            seq_names = [tup[1] for tup in self.cursor.execute('PRAGMA TABLE_INFO(Sequences)').fetchall()
                         if tup[1] not in ['sessio_id']]
            for seq in seq_names:
                for n in aux:
                    names.append("{0}_{1}".format(seq, n))
        names.append("label")
        return pd.DataFrame(new_data, columns=names)

    def calc_sequence_weight(self, session_id, ant_id):
        date1 = session_date(session_id)
        date2 = session_date(ant_id)
        days = (date1 - date2).days
        weight = 1/(days+1)
        return round(weight, 2)


class LmFitModels(ModelsGenerator):

    def _generate_models(self, args_dict):
        model_name = args_dict.get("lmfit_model", "GaussianModel")
        model_obj = getattr(mdl, model_name)
        all_sessions = self.loader.get_all_sessions()
        try:
            all_sessions = [x[0] for x
                            in self.loader.cursor.execute("SELECT id FROM LmFitModels WHERE id NOT IN ({0})"
                                                          .format(str(all_sessions)[1:-1])).fetchall()]
        except Exception as e:
            pass
        smodels = self.child_mgen.get_models(all_sessions, args_dict)
        res = []
        columns = None
        for i in range(len(smodels)):
            data = smodels.iloc[i, 1:-1]
            aux = self.fit_model(data, model_obj())
            if not aux:
                continue
            if not columns:
                columns = ['id'] + list(aux.keys())
            res.append([smodels.iloc[i, 0]] + list(aux.values()))
        new_df = pd.DataFrame(res, columns=columns)
        new_df.to_sql("LmFitModels", self.loader.db_connection, if_exists="append", index=False)

    def _get_models(self, session_ids, args_dict):
        return pd.read_sql(
            "SELECT LmFitModels.*, Sessions.label FROM Sessions INNER JOIN LmFitModels ON Sessions.id = LmFitModels.id"
            " WHERE Sessions.id IN ({0});".format(str(session_ids)[1:-1]), self.loader.db_connection)

    @staticmethod
    def fit_model(data, model):
        """
        :param
        data:   pandas.series with the histogram of the values occured in a session.
                The series has two unnamed columns:
                    * key: Glucose level value
                    * value: number of ocurences of the Glucose level value in this session

        model:  Model object from lmfit.models

        :return out:    Model object from lmfit.models constructed from the data passed
        """
        try:
            x = pd.Series([float(x) for x in data.index.values])
            y = pd.Series(data.values.tolist())
            pars = model.guess(y, x=x)
            out = model.fit(y, pars, x=x)
            return {par[0]: par[1].value for par in out.params.items()}
        except Exception as e:
            print(e)
            pass


class LmFitWordsModels(WordsModels):

    def _generate_models(self, args_dict):
        k = args_dict.get('k', 5)
        try:
            self.cursor.execute("DROP TABLE Words;")
        except:
            pass
        all_sessions = self.loader.get_all_sessions()
        models = self.child_mgen.get_models(all_sessions, args_dict)

        model_name = args_dict.get("lmfit_model", "GaussianModel")
        model_obj = getattr(mdl, model_name)
        res = []
        columns = None

        for lb_value, lb_rep in [(lb, str(lb)) for lb in [-1, 0, 1]]:
            fdp_data = models.ix[models.label == lb_value]
            kmeans, words = self.create_vocabulary(k, fdp_data)
            new = NamedTemporaryFile(mode="w")
            new.write('id;')
            for index in fdp_data.columns[1:-1]:
                new.write("n{0};".format(index).replace(".", "_"))
            new.write('label\n')
            i = 0
            for word in words:
                i += 1
                new.write(("w{0}_{1};".format(lb_rep, i)+str(word[:].round(10).tolist())[1:-1]+"\n").replace(',', ';')
                          .replace("-0.0;", "0.0;").replace("0.0;", "0;"))
            labels = kmeans.predict(fdp_data[fdp_data.columns[1:]])
            for i in range(len(labels)):
                self.cursor.execute(
                    "UPDATE Sessions SET word_id='w{0}_{1}' WHERE id='{2}';"
                        .format(lb_rep, labels[i]+1, fdp_data.iloc[i, 0])
                )
            new.flush()
            new.seek(0)
            df = pd.DataFrame.from_csv(new.name, sep=';', index_col=None)
            new.close()

            for i in range(len(df)):
                data = df.iloc[i, 1:-1]
                aux = self.fit_model(data, model_obj())
                if not aux:
                    continue
                if not columns:
                    columns = ['id'] + list(aux.keys()) + ['label']
                res.append([df.iloc[i, 0]] + list(aux.values()) + [df.iloc[i, -1]])

            new_df = pd.DataFrame(res, columns=columns)
            new_df.to_sql("Words", self.loader.db_connection, if_exists="append", index=False)

    @staticmethod
    def fit_model(data, model):
        """
        :param
        data:   pandas.series with the histogram of the values occured in a session.
                The series has two unnamed columns:
                    * key: Glucose level value
                    * value: number of ocurences of the Glucose level value in this session
    
        model:  Model object from lmfit.models
    
        :return out:    Model object from lmfit.models constructed from the data passed
        """
        try:
            x = pd.Series([float(x[1:].replace("_", ".")) for x in data.index.values])
            y = pd.Series(data.values.tolist())
            pars = model.guess(y, x=x)
            out = model.fit(y, pars, x=x)
            return {par[0]: par[1].value for par in out.params.items()}
        except Exception as e:
            print(e)
            pass

def myround(x, prec=3, base=.025):
    return round(base * round(float(x)/base), prec)


def session_date(sesion_name):
    """Takes a session id with format: {name}_{date}_{hour} and returns a date object"""
    name, date, hour = sesion_name.split('_')
    day, month, year = date.split('/')
    date_obj = datetime.strptime('{0}/{1}/{2} {3}'.format(day, month, year, hour), '%d/%m/%Y %H:%M')
    return date_obj


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    res = y[(int(window_len/2)-1):-(int(window_len/2))-1]
    return res