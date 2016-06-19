using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Globalization;

namespace OutParser
{
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }

        private void textBox1_DragDrop(object sender, DragEventArgs e)
        {
            string[] FileList = (string[])e.Data.GetData(DataFormats.FileDrop, false);
            tbDosya.Text = FileList[0];
            int counter = 0;

            try
            {
                string fileName = tbDosya.Text;
                StreamReader sr = new StreamReader(fileName, Encoding.Default);
                string path = Path.GetDirectoryName(fileName) + "\\";

                while (!sr.EndOfStream)
                {
                    StreamWriter sw = new StreamWriter(path + "card" + counter, false, Encoding.Default);
                    string str;
                    do
                    {
                        str = sr.ReadLine();
                        /*if (str.StartsWith("FN:"))
                            str = "FN;ENCODING=QUOTED-PRINTABLE:" + QuotedPrintableConverter.Encode(str.Substring(3));
                        else if (str.StartsWith("N:"))
                            str = "N;ENCODING=QUOTED-PRINTABLE:" + QuotedPrintableConverter.Encode(str.Substring(3));
                        else if (str.StartsWith("NICKNAME:"))
                            str = "NICKNAME;ENCODING=QUOTED-PRINTABLE:" + QuotedPrintableConverter.Encode(str.Substring(3));*/
                        sw.WriteLine(str);
                    }
                    while (str != "END:VCARD");
                    sw.Close();
                    counter++;
                }
            }
            catch (Exception ex)
            {
                toolStripStatusLabel1.Text = "Error: " + ex.ToString();
                return;
            }
        }

        private void textBox1_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.All;
            else
                e.Effect = DragDropEffects.None;
        }
    }

    public class QuotedPrintableConverter
    {
        private static string _Ascii7BitSigns;
        private const string _equalsSign = "=";
        private const string _defaultReplaceEqualSign = "=";

        /// <summary>
        /// Ctor.
        /// </summary>
        private QuotedPrintableConverter()
        {
            //do nothing
        }

        /// <summary>
        /// Encodes a not QP-Encoded string.
        /// </summary>
        /// <param name="value">The string which should be encoded.</param>
        /// <returns>The encoded string</returns>
        public static string Encode(string value)
        {
            return Encode(value, _defaultReplaceEqualSign);
        }

        /// <summary>
        /// Encodes a not QP-Encoded string.
        /// </summary>
        /// <param name="value">The string which should be encoded.</param>
        /// <param name="replaceEqualSign">The sign which should replace the "="-sign in front of 
        /// each QP-encoded sign.</param>
        /// <returns>The encoded string</returns>
        public static string Encode(string value, string replaceEqualSign)
        {
            //Alle nicht im Ascii-Zeichnsatz enthaltenen Zeichen werden ersetzt durch die hexadezimale 
            //Darstellung mit einem vorangestellten =
            //Bsp.: aus "ü" wird "=FC"
            //Bsp. mit Ersetzungszeichen "%"für das "=": aus "ü" wird "%FC"

            GetAllowedAsciiSigns();
            StringBuilder sb = new StringBuilder();
            foreach (char s in value)
            {
                if (_Ascii7BitSigns.LastIndexOf(s) > -1)
                    sb.Append(s);
                else
                {
                    int val = s;
                    string qp = Convert.ToString(val, 16);
                    qp = qp.Replace(_equalsSign, replaceEqualSign);
                    qp = _equalsSign + qp;
                    sb.Append(qp);
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Gets a string which contains the first 128-characters (ANSII 7 bit).
        /// </summary>
        private static void GetAllowedAsciiSigns()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 127; i++)
            {
                sb.Append(System.Convert.ToChar(i));
            }
            _Ascii7BitSigns = sb.ToString();
        }
    }
}
