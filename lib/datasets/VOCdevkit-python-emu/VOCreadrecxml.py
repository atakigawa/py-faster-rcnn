"""
python port of
VOCdevkit2007/VOCcode/VOCreadrecxml.m
"""

import xml.etree.ElementTree as ET
import os.path as osp


def voc_read_rec_xml(path):
    x = ET.parse(path).getroot()

    # rec=rmfield(x,'object');
    objects = x.findall('object')
    for elem in objects:
        x.remove(elem)
    rec = x

    # rec.size.width=str2double(rec.size.width);
    # rec.size.height=str2double(rec.size.height);
    # rec.size.depth=str2double(rec.size.depth);
    width = rec.find('size/width')
    width.text = float(width.text)
    height = rec.find('size/height')
    height.text = float(height.text)
    depth = rec.find('size/depth')
    depth.text = float(depth.text)

    # rec.segmented=strcmp(rec.segmented,'1');
    segmented = rec.find('segmented')
    segmented.text = segmented.text == '1'

    # rec.imgname=[x.folder '/JPEGImages/' x.filename];
    # rec.imgsize=str2double({x.size.width x.size.height x.size.depth});
    # rec.database=rec.source.database;
    imgname = ET.SubElement(rec, 'imgname')
    imgname.text = osp.join(
        rec.find('folder').text, 'JPEGImages', rec.find('filename').text)
    imgsize = ET.SubElement(rec, 'imgsize')
    imgsize.text = (width.text, height.text, depth.text)
    database = ET.SubElement(rec, 'database')
    database.text = rec.find('source').find('database').text

    # for i=1:length(x.object)
    #     rec.objects(i)=xmlobjtopas(x.object(i));
    # end
    elems = [_xmlobjtopas(elem, 'objects') for elem in objects]
    rec.extend(elems)

    return rec


def _xmlobjtopas(o, root_tag_name):
    p = ET.Element(root_tag_name)

    # p.class=o.name;
    elem = ET.SubElement(p, 'class')
    elem.text = o.find('name').text

    # if isfield(o,'pose')
    #     if strcmp(o.pose,'Unspecified')
    #         p.view='';
    #     else
    #         p.view=o.pose;
    #     end
    # else
    #     p.view='';
    # end
    elem = ET.SubElement(p, 'view')
    pose = o.find('pose')
    if pose is not None:
        if pose.text == 'Unspecified':
            elem.text = ''
        else:
            elem.text = pose.text
    else:
        elem.text = ''

    # if isfield(o,'truncated')
    #     p.truncated=strcmp(o.truncated,'1');
    # else
    #     p.truncated=false;
    # end
    elem = ET.SubElement(p, 'truncated')
    truncated = o.find('truncated')
    if truncated is not None:
        elem.text = truncated.text == '1'
    else:
        elem.text = False

    # if isfield(o,'difficult')
    #     p.difficult=strcmp(o.difficult,'1');
    # else
    #     p.difficult=false;
    # end
    elem = ET.SubElement(p, 'difficult')
    difficult = o.find('difficult')
    if difficult is not None:
        elem.text = difficult.text == '1'
    else:
        elem.text = False

    # p.label=['PAS' p.class p.view];
    # if p.truncated
    #     p.label=[p.label 'Trunc'];
    # end
    # if p.difficult
    #     p.label=[p.label 'Difficult'];
    # end
    elem = ET.SubElement(p, 'label')
    label_arr = ['PAS', p.find('class').text, p.find('view').text]
    if p.find('truncated').text:
        label_arr.append('Trunc')
    if p.find('difficult').text:
        label_arr.append('Difficult')
    elem.text = label_arr

    # p.orglabel=p.label;
    elem = ET.SubElement(p, 'orglabel')
    elem.text = p.find('label').text

    # p.bbox=str2double({o.bndbox.xmin o.bndbox.ymin o.bndbox.xmax o.bndbox.ymax}); # noqa
    # p.bndbox.xmin=str2double(o.bndbox.xmin);
    # p.bndbox.ymin=str2double(o.bndbox.ymin);
    # p.bndbox.xmax=str2double(o.bndbox.xmax);
    # p.bndbox.ymax=str2double(o.bndbox.ymax);
    xmin = float(o.find('bndbox/xmin').text)
    ymin = float(o.find('bndbox/ymin').text)
    xmax = float(o.find('bndbox/xmax').text)
    ymax = float(o.find('bndbox/ymax').text)

    elem = ET.SubElement(p, 'bbox')
    elem.text = (xmin, ymin, xmax, ymax)

    bndbox = ET.SubElement(p, 'bndbox')
    elem = ET.SubElement(bndbox, 'xmin')
    elem.text = xmin
    elem = ET.SubElement(bndbox, 'ymin')
    elem.text = ymin
    elem = ET.SubElement(bndbox, 'xmax')
    elem.text = xmax
    elem = ET.SubElement(bndbox, 'ymax')
    elem.text = ymax

    # if isfield(o,'polygon')
    #     warning('polygon unimplemented');
    #     p.polygon=[];
    # else
    #     p.polygon=[];
    # end
    elem = ET.SubElement(p, 'polygon')
    polygon = o.find('polygon')
    if polygon is not None:
        print 'warning: polygon unimplemented'
        elem.text = []
    else:
        elem.text = []

    # if isfield(o,'mask')
    #     warning('mask unimplemented');
    #     p.mask=[];
    # else
    #     p.mask=[];
    # end
    elem = ET.SubElement(p, 'mask')
    mask = o.find('mask')
    if mask is not None:
        print 'warning: mask unimplemented'
        elem.text = []
    else:
        elem.text = []

    # if isfield(o,'part')&&~isempty(o.part)
    #     p.hasparts=true;
    #     for i=1:length(o.part)
    #         p.part(i)=xmlobjtopas(o.part(i));
    #     end
    # else
    #     p.hasparts=false;
    #     p.part=[];
    # end
    parts = o.findall('part')
    if parts is not None and len(parts) > 0:
        elem = ET.SubElement(p, 'hasparts')
        elem.text = True
        elems = [_xmlobjtopas(el, 'part') for el in parts]
        p.extend(elems)
    else:
        elem = ET.SubElement(p, 'hasparts')
        elem.text = False
        elem = ET.SubElement(p, 'part')
        elem.text = []

    return p
