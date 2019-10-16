


def write_object(f, xmin, ymin, xmax, ymax):

	line = '\t<object>\n'
	f.writelines(line)

	line = '\t\t<name>seed</name>\n'
	f.writelines(line)

	line = '\t\t<pose>Unspecified</pose>\n'
	f.writelines(line)

	line = '\t\t<truncated>0</truncated>\n'
	f.writelines(line)

	line = '\t\t<difficult>0</difficult>\n'
	f.writelines(line)

	line = '\t\t<bndbox>\n'
	f.writelines(line)

	line = '\t\t\t<xmin>' + xmin + '</xmin>\n'
	f.writelines(line)

	line = '\t\t\t<ymin>' + ymin + '</ymin>\n'
	f.writelines(line)

	line = '\t\t\t<xmax>' + xmax + '</xmax>\n'
	f.writelines(line)

	line = '\t\t\t<ymax>' + ymax + '</ymax>\n'
	f.writelines(line)


	line = '\t\t</bndbox>\n'
	f.writelines(line)

	line = '\t</object>\n'
	f.writelines(line)

def write_xml(xml_fname, txt_fname):
	f = open('./xml/' + xml_fname,'w')

	line = '<annotation>\n'
	f.writelines(line)

	line = '\t<folder>gray</folder>\n'
	f.writelines(line)

	line = '\t<filename>' + txt_fname.split('.')[0] + '.jpg' + '</filename>\n'
	f.writelines(line)

	path = 'D:\\Hengyang\\Weekly_Report\\Week9_10292017\\Segmentation_improvement\\better-seed\\DeepSeedsDetection\\Train\\gray\\'
	line = '\t<path>' + path + txt_fname.split('.')[0] + '.jpg' + '</path>\n'
	f.writelines(line)

	line = '\t<source>\n'
	f.writelines(line)

	line = '\t\t<database>Unknown</database>\n'
	f.writelines(line)

	line = '\t</source>\n'
	f.writelines(line)

	line = '\t<size>\n'
	f.writelines(line)

	line = '\t\t<width>250</width>\n'
	f.writelines(line)

	line = '\t\t<height>200</height>\n'
	f.writelines(line)

	line = '\t\t<depth>3</depth>\n'
	f.writelines(line)

	line = '\t</size>\n'
	f.writelines(line)

	line = '\t<segmented>0</segmented>\n'
	f.writelines(line)

	# write the object information
	f_txt = open('./txt/' + txt_fname, 'r')
	lines = f_txt.readlines()
	f_txt.close()

	for line in lines:
		[ymin, xmin, ymax, xmax] = line.rstrip().split(',')
		write_object(f, xmin, ymin, xmax, ymax)

	line = '</annotation>\n'
	f.writelines(line)

	f.close()




if __name__ == '__main__':
	for i in range(1,1731):
		xml_fname = 'Tr_' + str(i) + '.xml'
		txt_fname = 'Tr_' + str(i) + '.txt'
		write_xml(xml_fname, txt_fname)
		print('process image ' + str(i) + ' ........')
