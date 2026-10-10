Pod::Spec.new do |s|
  s.name           = 'WoundDepth'
  s.version        = '1.0.0'
  s.summary        = 'Wound camera that measures the distance to the wound with ARKit'
  s.description    = 'An ARKit camera view: each photo is saved with the distance to the wound, which gives it a scale without a calibration sticker.'
  s.author         = ''
  s.homepage       = 'https://docs.expo.dev/modules/'
  s.platforms      = { :ios => '16.4' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.frameworks = 'ARKit', 'SceneKit', 'CoreImage', 'ImageIO'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_COMPILATION_MODE' => 'wholemodule'
  }

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
end
