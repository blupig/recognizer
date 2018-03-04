//
// yl-recognizer
// Copyright (C) 2017-2018 Yunzhu Li
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//

import UIKit
import AVFoundation

class MainViewController: UIViewController, UIImagePickerControllerDelegate, VideoFrameCaptureDelegate {

    @IBOutlet weak var viewCamera: UIView!
    @IBOutlet weak var lblCameraInfo: UILabel!
    @IBOutlet weak var imgTest: UIImageView!

    var vcc: VideoCaptureCoordinator?

    override func viewDidLoad() {
        super.viewDidLoad()

        // Video capture
        // Hide camera access info
        self.lblCameraInfo.isHidden = true

        vcc = VideoCaptureCoordinator(self)

        // Request for access
        vcc?.cameraAuth { (granted) in
            self.lblCameraInfo.isHidden = granted

            if !granted { return }

            if let error = self.vcc?.initCapture(previewView: self.viewCamera) {
                self.alert(title: "Error", message: error)
                return
            }

            // Start video capture
            self.vcc?.turnVideoCapture(on: true)
        }
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        vcc?.syncPreviewLayerSize(previewView: viewCamera)
    }

    override func viewWillDisappear(_ animated: Bool) {
        vcc?.turnVideoCapture(on: false)
    }

    @IBAction func btnPhotoLibraryAct(_ sender: UIBarButtonItem) {
        vcc?.captureNextFrame()
    }

    // UIAlertController convenience function
    func alert(title: String, message: String) {
        let controller = UIAlertController(title: title, message: message, preferredStyle: .alert)
        controller.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        self.present(controller, animated: true, completion: nil)
    }

    func frameCapture(didCapture image: UIImage) {
        print("UIImage captured")
        imgTest.image = image
    }
}
