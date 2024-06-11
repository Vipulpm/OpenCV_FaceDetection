using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Xml;
using static System.Net.Mime.MediaTypeNames;


namespace OpenCV_FaceDetection.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class FaceDetectionController : ControllerBase
    {
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
                System.Drawing.Image inputImage = System.Drawing.Image.FromStream(stream);
                Bitmap bitmap = new Bitmap(inputImage);

                // Convert Bitmap to Mat
                Mat img = BitmapToMat(bitmap);

                // Convert to grayscale
                Mat grayImg = new Mat();
                CvInvoke.CvtColor(img, grayImg, ColorConversion.Bgr2Gray);

                // Load the HaarCascade for face detection
                string haarCascadePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "haarcascades", "haarcascade_frontalface_default.xml");
                if (!System.IO.File.Exists(haarCascadePath))
                {
                    return BadRequest("Haar Cascade file not found.");
                }
                CascadeClassifier faceCascade = new CascadeClassifier(haarCascadePath);

                // Detect faces
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

                // Set the face count in the response header
                Response.Headers.Add("x-number-of-faces", facesDetected.Length.ToString());

                // Return the processed image
                return File(imageBytes, "image/jpeg");
            }
        }

        private Mat BitmapToMat(Bitmap bitmap)
        {
            // Ensure the bitmap is in the 24bppRgb format
            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                Bitmap temp = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                using (Graphics g = Graphics.FromImage(temp))
                {
                    g.DrawImage(bitmap, new Rectangle(0, 0, temp.Width, temp.Height));
                }
                bitmap = temp;
            }

            // Create a new Mat with the same size and type as the bitmap
            Mat mat = new Mat(bitmap.Height, bitmap.Width, DepthType.Cv8U, 3);

            // Lock the bitmap's bits
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

            // Copy the bitmap data to the Mat
            int stride = bitmapData.Stride;
            int bytes = stride * bitmap.Height;
            byte[] rgbValues = new byte[bytes];
            Marshal.Copy(bitmapData.Scan0, rgbValues, 0, bytes);
            bitmap.UnlockBits(bitmapData);

            // Set the Mat data
            Marshal.Copy(rgbValues, 0, mat.DataPointer, bytes);

            return mat;
        }

        private Bitmap MatToBitmap(Mat mat)
        {
            // Create a new Bitmap with the same size as the Mat
            Bitmap bitmap = new Bitmap(mat.Width, mat.Height, PixelFormat.Format24bppRgb);

            // Lock the bitmap's bits
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.WriteOnly, bitmap.PixelFormat);

            // Check if the Mat is of the expected type
            if (mat.Depth != DepthType.Cv8U || mat.NumberOfChannels != 3)
            {
                throw new NotSupportedException("Only 8-bit, 3-channel Mats are supported.");
            }

            // Copy the Mat data to the Bitmap
            int stride = bitmapData.Stride;
            int bytes = stride * mat.Height;
            byte[] rgbValues = new byte[bytes];
            Marshal.Copy(mat.DataPointer, rgbValues, 0, bytes);
            Marshal.Copy(rgbValues, 0, bitmapData.Scan0, bytes);
            bitmap.UnlockBits(bitmapData);

            return bitmap;
        }

    }
}
