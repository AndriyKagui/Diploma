   M S B A M L   `   `   `  ����M  IEmotionRecognition, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null7 clr-namespace:EmotionRecognitionEmotionRecognition  5   	   #    EmotionRecognition.MainWindow   5   	   W SPresentationCore, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=6.0.2.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5   	   R NSystem.Xaml, Version=6.0.2.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   2local clr-namespace:EmotionRecognition     5   	     � Title$  EmotionRecognition��5   	   $	��600��6$   $	��800��61   .��5	      �  � RowDefinitions 5
   
   �� 5      $
��Auto �6   �� 5      $��* �6   �� 5      $
��Auto �6   .+�5   
   �� -   5         ��Name$ videoSourceComboBox��5      $��10q�60   $
��Left=�6<   $	��250��6W   5      �� 5   
   -   5         ��Name$ cameraImage��5      $��1��6%   $��Uniform=�62   �� 5   
   $��2��6    �� Orientation$ 
Horizontal=�6"   $��Right=�6;   $��10q�6W   .+�5      �� -   5         ��Name$ emotionLabel��5      $��1��6.   $��10q�6;   $
��Left=�6G   $��Bottom=�6b   $
��Bold"�6}   $��16(�6�   �� 5      -   5         ��Name$ startButton��5      $��Start��6*   $��0,0,10,0q�6:   $	��100��6L   6X   �� 5      -   5        $ 
stopButton��5      $
��Stop��6)   $	��100��68   6D   5      5      5      