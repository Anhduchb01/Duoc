


def __remove_special_character(text):
	''' Remove special characters '''
	if (">>" in text):
		text = text.split('>>')[1]
	text = text.replace("(", " ")
	text = text.replace(")", " ")
	text = text.replace("/", " ")
	text = text.replace('\\', " ")
	text = text.replace(":", " ")
	text = text.replace(",", " ")
	text = text.replace("'"," ")
	text = text.replace('"'," ")
	return text

def split_sentence(text):
	''' Split big sentences to small sentences. Max sentences < 300 word '''
	text_process = []
	text = __remove_special_character(text)
	text_split_new_line = text.split('\n')
	text_split_by_dot = []
	for t in text_split_new_line:
		if (len(t.split()) > 400):
			text_split = t.split('.')
			text_split_by_dot += text_split
		else:
			text_split_by_dot.append(t)
	print(text_split_by_dot)
	print(len(text_split_by_dot))
	print(len(text_split_by_dot[1].split(' ')))
	join_text = ''
	for t in text_split_by_dot:
		if (len(t.split(' ')) > 5):
			if ((len(join_text.split(' ')) + len(t.split(' '))) < 400):
				join_text += t
				
			else:
				text_process.append(join_text)
				join_text = ''
	if (join_text):
		text_process.append(join_text)
	return text_process
text = """CEO GSM hiện đang thông báo tuyển dụng vị trí Giám đốc vận hành xe máy điện (GreenBike\/GreenExpress).
							['  Mới đây, CEO công ty GSM - Nguyễn Văn Thanh đã có bài chia sẻ về khả năng mở dịch vụ xe ôm công nghệ bằng xe máy điện. ', '  Theo vị CEO trẻ, Việt Nam đứng thứ 2 trên thế giới về tỉ lệ sở hữu xe máy, ước tính hơn 42 triệu chiếc (bao gồm cả những xe không còn lưu hành). Mỗi năm thị trường Việt Nam lại tiêu thụ thêm 3 triệu xe và phần lớn trong số đó là xe máy xăng. ', '  Việc chuyển sang sử dụng xe máy điện sẽ chấm dứt việc xả khói đen, giảm tiếng ồn.\xa0Tiếp theo là tiết kiệm, chi phí để nuôi một chiếc xe máy điện sẽ tốt hơn nhiều so với một chiếc xe máy xăng. ', '  Cuối cùng là trải nghiệm, chắc hẳn khách hàng sẽ thấy thoải mái và thích thú hơn khi ngồi trên một chiếc xe máy điện, đặc biệt đó lại là sản phẩm và trí óc của người Việt Nam. \ufeff ', '  ', '  , CEO Nguyễn Văn Thanh đặt câu hỏi. \ufeff ', '  Dù chưa có câu trả lời chính thức nhưng CEO GSM cũng đồng thời thông báo tuyển dụng vị trí Giám đốc vận hành xe máy điện (GreenBike\/GreenExpress). \ufeff ', '  Hiện nay,\xa0GSM đã đưa vào hàng ngàn xe tại Hà Nội, Hồ Chí Minh và ngay trong tháng 5 này sẽ có mặt tại Huế, tháng 6 sẽ tiếp tục tại Nha Trang, Đà Nẵng.\xa0Nguyễn Văn Thanh đã tiết lộ rằng cuối tháng sẽ ra thêm 1.000 xe ở Hà Nội, ngoài ra, số lượng xe trong Thành phố Hồ Chí Minh trong tháng này cũng sẽ tăng thêm.', ' ']"""
text_process = split_sentence(text)
print(text_process)
print(len(text_process))
print(len(text.split(" ")))
def group_entities(entities):
	grouped_entities = []
	current_entity = None

	for entity in entities:
		if entity['entity'].startswith('B-'):
			if current_entity:
				grouped_entities.append(current_entity)
			current_entity = {'type': entity['entity'][2:], 'start': entity['start'], 'end': entity['end'], 'name': entity['word']}
		elif entity['entity'].startswith('I-'):
			if current_entity and current_entity['type'] == entity['entity'][2:]:
				current_entity['end'] = entity['end']
				current_entity['name'] += " "+ entity['word']
			else:
				if current_entity:
					grouped_entities.append(current_entity)
				current_entity = None

	if current_entity:
		grouped_entities.append(current_entity)

	return grouped_entities
from underthesea import ner
def generate_lib_tag(text):
	entities = ner(text, deep=True)
	grouped_entities = group_entities(entities)
	return grouped_entities
tags_array = []
tags_name_array = []
for sentence in text_process:
	tags = generate_lib_tag(sentence)
	for tag in tags:
		if tag["name"] not in tags_name_array:
			tags_array.append(tag)
			tags_name_array.append(tag["name"])
	for tag in tags_array:
		for tag1 in tags_array:
			if tag["name"] in tag1["name"] and len(tag["name"]) < len(tag1["name"]):
				tags_array.remove(tag1)
print("Final")
print(tags_array)
print(len(tags_array))