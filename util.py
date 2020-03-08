from imports import *

class PreProcess():

    def __init__(self, path):

        self.path = path
        self.dataBenign = {"permissions":[], "isValidAPK":[], "services":[], "receivers":[] }
        self.dataMalign = {"permissions":[], "isValidAPK":[], "services":[], "receivers":[] }

        # Placeholders for the massive vocabulary that will be generated during pre processing.
        self.vocabPerm = list()
        self.vocabServ = list()
        self.vocabRecv = list()

        if LOAD_DATA:

            print("Loading data..")
            f = open("pickled/dataDictBenign.pkl", "rb")
            self.dataBenign = pickle.load(f)
            f.close()

            f = open("pickled/dataDictMalign.pkl", "rb")
            self.dataMalign = pickle.load(f)
            f.close()

            f = open("pickled/vocabPerm.pkl", "rb")
            self.vocabPerm = pickle.load(f)
            f.close()
            f = open("pickled/vocabServ.pkl", "rb")
            self.vocabServ = pickle.load(f)
            f.close()

            f = open("pickled/vocabRecv.pkl", "rb")
            self.vocabRecv = pickle.load(f)
            f.close()

            f = open("pickled/vocabLen.pkl", "rb")
            self.vocabLengths = pickle.load(f)
            f.close()


        elif MAKE_DICT:

            # Extract info from APKS and store them in dict with the feature identifier as keys.
            # The method makeDataDicts which takes care of this is called here.
            print("Extracting data from APKs..")
            self.makeDataDicts()

            print("Pickling the data dicts and vocabs...")

            f = open("pickled/dataDictBenign.pkl", "wb")
            pickle.dump(self.dataBenign, f)
            f.close()

            f = open("pickled/dataDictMalign.pkl", "wb")
            pickle.dump(self.dataMalign, f)
            f.close()

            f = open("pickled/vocabPerm.pkl", "wb")
            pickle.dump(self.vocabPerm, f)
            f.close()

            f = open("pickled/vocabRecv.pkl", "wb")
            pickle.dump(self.vocabRecv, f)
            f.close()

            f = open("pickled/vocabServ.pkl", "wb")
            pickle.dump(self.vocabServ, f)
            f.close()

            # A dict that stores lengths os all vocabularies used.
            self.vocabLengths = {"perm":len(self.vocabPerm), "serv":len(self.vocabServ), \
                            "recv":len(self.vocabRecv)}

            f = open("pickled/vocabLen.pkl", 'wb')
            pickle.dump(self.vocabLengths, f)
            f.close()


        if LOAD_FEATS:

            print("Loading Features.")
            f = open("pickled/feats.pkl", "rb")
            self.feats = pickle.load(f)
            f.close()

            f = open("pickled/labels.pkl", "rb")
            self.labels = pickle.load(f)
            f.close()

        else:

            # Convert the data stored in dict to a appropriate integer dataframe in pandas
            # the method makeDataFrames is called here.
            print("Creating Data Frames..")
            self.makeDataFrames()

            f = open("pickled/feats.pkl", "wb")
            pickle.dump(self.feats, f)
            f.close()

            f = open("pickled/labels.pkl", "wb")
            pickle.dump(self.labels, f)
            f.close()

        # print "-------------------------------------------------------------------------------------------------------"

    def getFeats(self):
        return (self.feats, self.labels)

    def makeDataDicts(self):

        try:
            bt = dt.now()
            # Creating the benign data dict first.
            print("Processing Benign Folder")
            count = 1
            for filename in glob.glob(self.path + "Benign/*"):

                try:
                    t = dt.now()
                    a, d, dx = AnalyzeAPK(filename)
                    print(count, "--", os.path.basename(filename), "--", str(dt.now() - t))

                    # A temporary list to hold all tentative features extracted from a APK
                    temp = list()
                    temp.append(int(a.valid_apk))
                    temp.append(a.get_permissions())
                    temp.append(a.get_services())
                    temp.append(a.get_receivers())

                except Exception as  e:
                    print("Error during analysis of above APK - IGNORED")
                    print(e.message)
                    continue

                # Checking whether the APK is valid according to Androguard
                self.dataBenign["isValidAPK"].append(temp[0])

                print("Extracting app permissions")
                # Adding app permissions as a multi valued attribute
                if temp[1]:

                    perm = list()
                    for p in temp[1]:
                        if p not in self.vocabPerm:
                            self.vocabPerm.append(p)
                            perm.append(self.vocabPerm.index(p))
                        else:
                            perm.append(self.vocabPerm.index(p))

                    self.dataBenign["permissions"].append(perm)
                else:
                    self.dataBenign["permissions"].append(list())

                print("Extracting app services")
                # Adding app services as a multi valued attribute
                if temp[2]:

                    serv = list()
                    for p in temp[2]:
                        if p not in self.vocabServ:
                            self.vocabServ.append(p)
                            serv.append(self.vocabServ.index(p))
                        else:
                            serv.append(self.vocabServ.index(p))

                    self.dataBenign["services"].append(serv)
                else:
                    self.dataBenign["services"].append(list())

                print("Extracting app recievers")
                # Adding app receivers as a multi valued attribute
                if temp[3]:

                    recv = list()
                    for p in temp[3]:
                        if p not in self.vocabRecv:
                            self.vocabRecv.append(p)
                            recv.append(self.vocabRecv.index(p))
                        else:
                            recv.append(self.vocabRecv.index(p))

                    self.dataBenign["receivers"].append(recv)
                else:
                    self.dataBenign["receivers"].append(list())

                count += 1
                if count == MAX_DATA / 2:
                    break
            print("Total time taken to process benign folder -- ", str(dt.now() - bt))

        except Exception as e:
            print(e.message)
            exit(0)

        try:
            mt = dt.now()
            print("Processing Malign Folder")
            count = 1
            for filename in glob.glob(self.path + "Malign/*"):

                try:
                    t = dt.now()
                    a, d, dx = AnalyzeAPK(filename)
                    print(count, "--", os.path.basename(filename), "--", str(dt.now() - t))

                    temp = list()
                    temp.append(int(a.valid_apk))
                    temp.append(a.get_permissions())
                    temp.append(a.get_services())
                    temp.append(a.get_receivers())
                except Exception as  e:

                    print("Error during analysis of above APK - IGNORED")
                    print(e.message)
                    continue

                self.dataMalign["isValidAPK"].append(temp[0])

                print("Extracting app permissions")
                # Adding app permissions as a multi valued attribute
                if temp[1]:

                    perm = list()
                    for p in temp[1]:
                        if p not in self.vocabPerm:
                            self.vocabPerm.append(p)
                            perm.append(self.vocabPerm.index(p))
                        else:
                            perm.append(self.vocabPerm.index(p))

                    self.dataMalign["permissions"].append(perm)
                else:
                    self.dataMalign["permissions"].append(list())

                print("Extracting app services")
                # Adding app services as a multi valued attribute
                if temp[2]:

                    serv = list()
                    for p in temp[2]:
                        if p not in self.vocabServ:
                            self.vocabServ.append(p)
                            serv.append(self.vocabServ.index(p))
                        else:
                            serv.append(self.vocabServ.index(p))

                    self.dataMalign["services"].append(serv)
                else:
                    self.dataMalign["services"].append(list())

                print("Extracting app receivers")
                # Adding app receivers as a multi valued attribute
                if temp:

                    recv = list()
                    for p in temp[3]:
                        if p not in self.vocabRecv:
                            self.vocabRecv.append(p)
                            recv.append(self.vocabRecv.index(p))
                        else:
                            recv.append(self.vocabRecv.index(p))

                    self.dataMalign["receivers"].append(recv)
                else:
                    self.dataMalign["receivers"].append(list())

                count += 1
                if count == MAX_DATA / 2:
                    break
            print("Total time taken to process malign folder -- ", str(dt.now() - mt))
        except Exception as e:
            print(e.message)
            exit(0)

    def makeDataFrames(self):

        try:
            # Initialize a empty dataframe for benign APKs with index in range(1,dim(feat_space)+1)
            data_len = len(self.dataBenign["isValidAPK"])+1
            index = np.array(range(1, data_len ))
            self.dfBenign = pd.DataFrame(index = index)

            # Converting isValid list to a panda series and appending it to the data frame.
            isValid = pd.Series(self.dataBenign['isValidAPK'], name="isValidAPK", index = index)
            self.dfBenign[isValid.name] = isValid

            # Converting multi valued permission attributes into a huge hot vector, further
            # making a data frame out of it and joining it with the main data frame
            perm = self.makeHotMatrix(self.dataBenign["permissions"], self.vocabLengths["perm"])
            columns = self.makeContCol("perm", self.vocabLengths["perm"])
            temp = pd.DataFrame(perm, columns=columns, index=index)
            self.dfBenign = self.dfBenign.join(temp)

            # Converting multi valued service attributes into a huge hot matrix, further
            # making a data frame out of it and joining it with the main data frame
            serv = self.makeHotMatrix(self.dataBenign["services"], self.vocabLengths["serv"])
            columns = self.makeContCol("serv", self.vocabLengths["serv"])
            temp = pd.DataFrame(serv, columns=columns, index=index)
            self.dfBenign = self.dfBenign.join(temp)

            # Converting multi valued reciever attributes into a huge hot vector, further
            # making a data frame out of it and joining it with the main data frame
            recv = self.makeHotMatrix(self.dataBenign["receivers"], self.vocabLengths["recv"])
            columns = self.makeContCol("recv", self.vocabLengths["recv"])
            temp = pd.DataFrame(recv, columns=columns, index=index)
            self.dfBenign = self.dfBenign.join(temp)

        except Exception as e:
            print("Error while creating benign dataframe")
            print(e.args, e.message)
            exit(0)

        try:
            # Initialize a empty dataframe for malign with index in range(dim(feat_space)+1, 2*dim(feat_space+1)
            index = np.array(range(data_len, 2*data_len-1))
            self.dfMalign = pd.DataFrame(index=index)

            isValid = pd.Series(self.dataMalign['isValidAPK'], name="isValidAPK", index=index)
            self.dfMalign[isValid.name] = isValid

            perm = self.makeHotMatrix(self.dataMalign["permissions"], self.vocabLengths["perm"])
            columns = self.makeContCol("perm", self.vocabLengths["perm"])
            temp = pd.DataFrame(perm, columns=columns, index=index)
            self.dfMalign = self.dfMalign.join(temp)

            serv = self.makeHotMatrix(self.dataMalign["services"], self.vocabLengths["serv"])
            columns = self.makeContCol("serv", self.vocabLengths["serv"])
            temp = pd.DataFrame(serv, columns=columns, index=index)
            self.dfMalign = self.dfMalign.join(temp)

            recv = self.makeHotMatrix(self.dataMalign["receivers"], self.vocabLengths["recv"])
            columns = self.makeContCol("recv", self.vocabLengths["recv"])
            temp = pd.DataFrame(recv, columns=columns, index=index)
            self.dfMalign = self.dfMalign.join(temp)

        except Exception as e:
            print("Error while creating malign dataframe")
            print(e.args, e.message)
            exit(0)

        # Storing the benign and malign feature set into CSV
        self.dfBenign.to_csv("csv/benign.csv", index=False)

        self.dfMalign.to_csv("csv/malign.csv", index=False)

        # Creating the final label set and feature set.
        self.labels =  np.array([0]*self.dfBenign.shape[0] + [1]*self.dfMalign.shape[0])
        self.feats = pd.concat([self.dfBenign, self.dfMalign])


    def makeHotMatrix(self, vec2D, len):

        hotMat = list()
        for vec in vec2D:
            if vec:
                hotMat.append(self.makeHotVector(vec, len))
            else:
                hotMat.append(np.zeros(len, dtype='int'))
        return hotMat

    def makeHotVector(self, vec, len):

        hotVec = np.zeros(len, dtype='int')
        hotVec[vec] = 1

        return hotVec

    def makeContCol(self, base, len):

        col = list()
        for i in range(len):
            col.append(base + "_" + str(i+1))

        return col




