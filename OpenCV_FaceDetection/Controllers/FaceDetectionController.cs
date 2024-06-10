using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Xml;


namespace OpenCV_FaceDetection.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class FaceDetectionController : ControllerBase
    {
        /*[HttpPost("detect")]
        public IActionResult DetectFaces(IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
                return BadRequest("No image file provided.");

            using (var stream = new MemoryStream())
            {
                imageFile.CopyTo(stream);
                stream.Position = 0;

                // Load the image from the stream
                System.Drawing.Image inputImage = System.Drawing.Image.FromStream(stream);
                Bitmap bitmap = new Bitmap(inputImage);

                // Convert Bitmap to Mat
                Mat img = BitmapToMat(bitmap);

                // Convert to grayscale
                Mat grayImg = new Mat();
                CvInvoke.CvtColor(img, grayImg, ColorConversion.Bgr2Gray);

                // Load the HaarCascade for face detection
                string haarCascadePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "haarcascade_frontalface_default.xml");
                CascadeClassifier faceCascade = new CascadeClassifier(haarCascadePath);

                // Detect faces
                Rectangle[] facesDetected = faceCascade.DetectMultiScale(
                    grayImg,
                    1.1,
                    10,
                    new Size(20, 20),
                    Size.Empty);

                // Draw rectangles around detected faces
                foreach (Rectangle face in facesDetected)
                {
                    CvInvoke.Rectangle(img, face, new MCvScalar(255, 0, 0), 2);
                }

                // Convert the Mat back to a byte array
                byte[] imageBytes;
                using (var ms = new MemoryStream())
                {
                    Bitmap resultBitmap = MatToBitmap(img);
                    resultBitmap.Save(ms, ImageFormat.Jpeg);
                    imageBytes = ms.ToArray();
                }

                // Return the processed image
                return File(imageBytes, "image/jpeg");
            }
        }

        private Mat BitmapToMat(Bitmap bitmap)
        {
            Mat mat = new Mat();
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

            // Create a new byte array to store the image data
            byte[] imageData = new byte[bitmapData.Stride * bitmapData.Height];
            Marshal.Copy(bitmapData.Scan0, imageData, 0, imageData.Length);

            // Create a Mat using the image data
            mat = new Mat(bitmap.Height, bitmap.Width, DepthType.Cv8U, 3);
            mat.SetTo(imageData);

            bitmap.UnlockBits(bitmapData);
            return mat;
        }

        private Bitmap MatToBitmap(Mat mat)
        {
            // Convert Mat to byte array
            byte[] data = new byte[mat.Rows * mat.Cols * mat.ElementSize];
            Marshal.Copy(mat.DataPointer, data, 0, data.Length);

            // Create bitmap from byte array
            Bitmap bitmap = new Bitmap(mat.Cols, mat.Rows, PixelFormat.Format24bppRgb);
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, mat.Cols, mat.Rows), ImageLockMode.WriteOnly, bitmap.PixelFormat);
            Marshal.Copy(data, 0, bitmapData.Scan0, data.Length);
            bitmap.UnlockBits(bitmapData);

            return bitmap;
        }*/

        [HttpPost("detect")]
        public IActionResult DetectFaces(IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
                return BadRequest("No image file provided.");

            using (var stream = new MemoryStream())
            {
                imageFile.CopyTo(stream);
                stream.Position = 0;

                // Load the image from the stream
                // Convert Bitmap to Emgu.CV Image<Bgr, byte>
                using (Image<Bgr, byte> img = MatToBitmap().ToImage<Bgr, byte>())
                {
                    // Convert to grayscale
                    using (Image<Gray, byte> grayImg = img.Convert<Gray, byte>())
                    {
                        // Load the HaarCascade for face detection
                        string haarCascadePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "haarcascade_frontalface_default.xml");

                        // Load the XML document
                        XmlDocument doc = new XmlDocument();
                        doc.Load(haarCascadePath);

                        // Parse XML to get cascade data
                        XmlNode cascadeNode = doc.SelectSingleNode("//opencv_storage/cascade");
                        string cascadeData = cascadeNode.InnerXml;

                        // Create a cascade classifier with the cascade data
                        using (var faceCascade = new CascadeClassifier())
                        {
                            faceCascade.LoadFromString(cascadeData);

                            // Detect faces
                            Rectangle[] facesDetected = faceCascade.DetectMultiScale(
                                grayImg,
                                1.1,
                                10,
                                new Size(20, 20),
                                Size.Empty);

                            // Draw rectangles around detected faces
                            foreach (Rectangle face in facesDetected)
                            {
                                img.Draw(face, new Bgr(Color.Red), 2);
                            }

                            // Convert the image back to byte array
                            byte[] imageBytes = img.ToJpegData();

                            // Return the processed image
                            return File(imageBytes, "image/jpeg");
                        }
                    }
                }
            }
        }
        private Mat BitmapToMat(Bitmap bitmap)
        {
            Mat mat = new Mat();
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

            // Create a new byte array to store the image data
            byte[] imageData = new byte[bitmapData.Stride * bitmapData.Height];
            Marshal.Copy(bitmapData.Scan0, imageData, 0, imageData.Length);

            // Create a Mat using the image data
            mat = new Mat(bitmap.Height, bitmap.Width, DepthType.Cv8U, 3);
            mat.SetTo(imageData);

            bitmap.UnlockBits(bitmapData);
            return mat;
        }

        private Bitmap MatToBitmap(Mat mat)
        {
            // Convert Mat to byte array
            byte[] data = new byte[mat.Rows * mat.Cols * mat.ElementSize];
            Marshal.Copy(mat.DataPointer, data, 0, data.Length);

            // Create bitmap from byte array
            Bitmap bitmap = new Bitmap(mat.Cols, mat.Rows, PixelFormat.Format24bppRgb);
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, mat.Cols, mat.Rows), ImageLockMode.WriteOnly, bitmap.PixelFormat);
            Marshal.Copy(data, 0, bitmapData.Scan0, data.Length);
            bitmap.UnlockBits(bitmapData);

            return bitmap;
        }
    }
}
