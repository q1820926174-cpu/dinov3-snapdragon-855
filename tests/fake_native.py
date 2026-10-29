"""Protocol-only test double; never used for hardware or performance claims."""
import os,struct,sys,time
import numpy as np
sys.stdout.buffer.write(b'RDY1');sys.stdout.buffer.flush()
count=0
while True:
    data=sys.stdin.buffer.read(224*224*3)
    if not data:break
    if len(data)!=224*224*3:raise SystemExit(20)
    count+=1
    if count>3:
        if os.getenv('DINO_TEST_CRASH'):raise SystemExit(17)
        time.sleep(float(os.getenv('DINO_TEST_DELAY','0')))
    vector=np.zeros(768,dtype='<f4');vector[0]=data[0]+1;vector[1]=256-data[0]
    sys.stdout.buffer.write(vector.tobytes()+struct.pack('<d',1.0));sys.stdout.buffer.flush()
