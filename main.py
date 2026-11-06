from voxelform.server import app
if __name__=='__main__':
    import os,uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('VOXELFORM_PORT','8010')))
