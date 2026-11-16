from pathlib import Path
import json
import sys
from PIL import Image,ImageDraw


def main():
    output=Path(sys.argv[1] if len(sys.argv)>1 else 'data/smoke').resolve();output.mkdir(parents=True,exist_ok=True)
    samples=[]
    for index in range(2):
        paths=[]
        for modality,color in [('rgb',(220,160,80)),('ir',(220,220,220))]:
            image=Image.new('RGB',(32,32),(20,20,20));draw=ImageDraw.Draw(image)
            draw.rectangle((12+index,12,20+index,18),fill=color)
            path=output/f'{modality}_{index}.png';image.save(path);paths.append(str(path))
        annotation=[{'category_id':0,'bbox':[16+index,15,8,6,0.]}]
        samples.append({'id':str(index),'rgb':paths[0],'ir':paths[1],'annotations':annotation,'ir_annotations':annotation})
    (output/'train.json').write_text(json.dumps({'classes':['object'],'samples':samples},indent=2),encoding='utf-8')
    print(output/'train.json')


if __name__=='__main__':main()
