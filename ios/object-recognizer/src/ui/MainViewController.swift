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

class MainViewController: UIViewController, UIImagePickerControllerDelegate {

    @IBOutlet weak var viewCamera: UIView!
    @IBOutlet weak var lblCameraInfo: UILabel!
    private var cameraAuthorized: Bool = false

    override func viewDidLoad() {
        super.viewDidLoad()

        // Camera authorization
        cameraAuth()
    }

    private func cameraAuth() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            // Already authorized
            cameraAuthorized = true
            lblCameraInfo.isHidden = true
            break
        case .notDetermined:
            // Need to ask, hide label first
            lblCameraInfo.isHidden = true
            AVCaptureDevice.requestAccess(for: .video, completionHandler: { (granted) in
                self.cameraAuthorized = granted
                DispatchQueue.main.async {
                    self.lblCameraInfo.isHidden = granted
                }

                if granted {

                }
            })
            break
        default:
            // Previously denied access
            cameraAuthorized = false
            lblCameraInfo.isHidden = false
            break
        }
    }
}
