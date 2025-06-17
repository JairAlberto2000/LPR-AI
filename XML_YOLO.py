import os
import xml.etree.ElementTree as ET

def xml_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)
    
    yolo_lines = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower()
        if class_name in ["licence", "license_plate"]:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if yolo_lines:
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")
        with open(output_path, "w") as f:
            f.write("\n".join(yolo_lines))

xml_dir = "C:/Users/Alex/Downloads/archive/annotations"
output_dir = "C:/Users/Alex/Downloads/archive/labels"

os.makedirs(output_dir, exist_ok=True)
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        xml_to_yolo(os.path.join(xml_dir, xml_file), output_dir)
print("¡Conversión completada! Revisa que los .txt no estén vacíos.")