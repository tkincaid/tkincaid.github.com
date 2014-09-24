class MockFileStorage():
    ''' Mocks Flask's FileStorage class. From the documentation:
        The FileStorage class is a thin wrapper over incoming files. It is used by the request object to represent uploaded files.
        All the attributes of the wrapper stream are proxied by the filee storagee so it is possible to do storage.read() instead of the long form storage.stream.read()

        http://werkzeug.pocoo.org/docs/datastructures/#werkzeug.datastructures.FileStorage
    '''
    def __init__(self, filename):
        self.filename =filename
        self.save_called = False

    def save(self, new_filename):
        open(new_filename, 'w')
        self.save_called = True
