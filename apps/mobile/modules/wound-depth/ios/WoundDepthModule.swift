import ARKit
import ExpoModulesCore

public class WoundDepthModule: Module {
  public func definition() -> ModuleDefinition {
    Name("WoundDepth")

    // False on the simulator and on devices without ARKit world tracking: the app then uses its ordinary camera.
    Constant("isSupported") {
      ARWorldTrackingConfiguration.isSupported
    }

    // True on phones with a LiDAR sensor (iPhone Pro): the distance is then read directly, without moving the phone.
    Constant("hasDepthSensor") {
      ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    View(WoundDepthView.self) {
      Events("onDistance")

      AsyncFunction("capture") { (view: WoundDepthView) -> [String: Any] in
        try view.capture()
      }
    }
  }
}
