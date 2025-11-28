export interface AppConfig {
  pageTitle: string;
  pageDescription: string;
  companyName: string;

  supportsChatInput: boolean;
  supportsVideoInput: boolean;
  supportsScreenShare: boolean;
  isPreConnectBufferEnabled: boolean;

  logo: string;
  startButtonText: string;
  accent?: string;
  logoDark?: string;
  accentDark?: string;

  // for LiveKit Cloud Sandbox
  sandboxId?: string;
  agentName?: string;
}

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Flipkart',
  pageTitle: 'FlipFresh',
  pageDescription: 'Your personal shopping assistant powered by AI.',

  supportsChatInput: true,
  supportsVideoInput: true,
  supportsScreenShare: true,
  isPreConnectBufferEnabled: true,

  logo: '/logo1.png',
  accent: '#002cf2',
  logoDark: '/logo1.png',
  accentDark: '#ffffff',
  startButtonText: 'Order Now',

  // for LiveKit Cloud Sandbox
  sandboxId: undefined,
  agentName: undefined,
};
