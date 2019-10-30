#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
身份证文字+数字生成类


"""
import numpy as np
import freetype
import copy
import random
import matplotlib.pylab as plt
#from captcha.image import ImageCaptcha
from PIL import Image

#返回可以产生给定形状，位置，内部文字内容、大小、颜色的图片(数组类型变量)的对象实例，对freetype的再封装
class put_chinese_text(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        '''
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        '''
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender/64.0

        #descender = metrics.descender/64.0
        #height = metrics.height/64.0
        #linegap = height - ascender + descender
        ypos = int(ascender)

        #text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1]+ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        '''
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        '''
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6   # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale)*0x10000, int(0.2*0x10000),\
                                 int(0.0*0x10000), int(1.1*0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        '''
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        '''
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row*cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]


class gen_id_card(object):
    def __init__(self):
       #self.words = open('AllWords.txt', 'r').read().split(' ')
       self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
       self.char_set = self.number #字符集
       #self.char_set = self.words + self.number
       self.len = len(self.char_set)#字符集的字符种类数
       
       self.max_size = 18#可以随机产生的字符的最大数量
       self.ft = put_chinese_text('fonts/DejaVuSansMono-Bold.ttf')
       
    #随机生成字串，长度固定
    #返回text,及对应的向量
    def gen_text(self, is_ran=False):
        text = ''
        vecs = np.zeros((self.max_size * self.len))

        #唯一变化，随机设定长度
        if is_ran == True:
            size = random.randint(1, self.max_size)
        else:
            size = self.max_size

        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i * self.len:(i + 1) * self.len] = np.copy(vec)
        return text, vecs

    '''
    gen_image()方法返回
    
    image_data：图片像素数据 (32,256)
    
    label： 图片标签 18位数字字符 
    
    vec :  图片标签转成向量表示 (180,)  代表每个数字所处的列，总长度 18 * 10
    
    '''
    
    # 根据生成的text，生成image,返回标签和图片元素数据
    def gen_image(self,is_ran=False):
        text,vec = self.gen_text(is_ran)#调用方法得到随机生成的字串和其张量形式
        img = np.zeros([32,256,3])#每个图片的张量形状
        color_ = (255,255,255) #   RGB-黑色
        pos = (0, 0)
        text_size = 21
        image = self.ft.draw_text(img, pos, text, text_size, color_)
        # 仅返回单通道值，颜色对于汉字识别没有什么意义
        return image[:,:,2],text,vec

    #单字转向量
    def char2vec(self, c):
        vec = np.zeros((self.len))
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec
        
    #向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if(vecs[i] == 1):
                text = text + self.char_set[i % self.len]
        return text

if __name__ == '__main__':
    genObj = gen_id_card()

    for i in range(5):
        image_data, label, vec = genObj.gen_image(True)
        plt.imshow(image_data)
        plt.show()
        plt.figure()
    # print(type(image_data),image_data.shape)#图片的形式是一个二维矩阵
    # print(type(label),label)
    # print(type(vec), vec)




