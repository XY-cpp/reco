import json
from PIL import Image
from PIL import ImageDraw
from collections import namedtuple
from cityscapesscripts.helpers.annotation import Annotation

Label = namedtuple( 'Label' , ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'lip'                  ,  1 ,        0 , 'object'          , 1       , True         , False        , (  0, 80,100) ),
    Label(  'tongue'               ,  2 ,        1 , 'object'          , 1       , True         , False        , (  0,  0,230) ),
]
name2label      = { label.name    : label for label in labels           }

def createLabelImage(annotation, encoding, outline=None):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )
    # the background
    if encoding == "ids":
        background = name2label['unlabeled'].id
    elif encoding == "trainIds":
        background = name2label['unlabeled'].trainId
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( labelImg )

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        if not label in name2label:
            print( "Label '{}' not known.".format(label) )

        # If the ID is negative that polygon should not be drawn
        if name2label[label].id < 0:
            continue

        if encoding == "ids":
            val = name2label[label].id
        elif encoding == "trainIds":
            val = name2label[label].trainId
        elif encoding == "color":
            val = name2label[label].color

        try:
            if outline:
                drawer.polygon( polygon, fill=val, outline=outline )
            else:
                drawer.polygon( polygon, fill=val )
        except:
            print("Failed to draw polygon with label {}".format(label))
            raise

    return labelImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the label image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
#     - "color"    : classes are encoded using the corresponding colors
def cs_json2cs_img(inJson,encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonText(json.dumps(inJson))
    labelImg = createLabelImage( annotation , encoding )
    return labelImg

def labelme_json2cs_json(labelme_json:str):
    cs_json = {}
    objects = []
    num = -1
    num = num + 1
    with open(labelme_json) as f:
        data = json.load(f)
        cs_json['imgHeight'] = data['imageHeight']
        cs_json['imgWidth'] = data['imageWidth']
        for shapes in data['shapes']:
            obj = {}
            label = shapes['label']
            obj['label'] = label
            points = shapes['points']
            p_type = shapes['shape_type']
            if p_type == 'polygon':
                obj['polygon'] = points
            objects.append(obj)
        cs_json['objects'] = objects
    return cs_json

def json2labelImg(labelme_json):
    cs_json = labelme_json2cs_json(labelme_json)
    cs_img = cs_json2cs_img(cs_json)
    return cs_img

# if __name__ == "__main__":
    # json2labelImg("dataset/sick_test/DSC_5032-461053-1-73.json").save("a.png")
